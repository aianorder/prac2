"""Behavior cloning: collect a rule teacher's actions, fit a student, evaluate both."""

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import joblib
import matplotlib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from practice_io import new_directory, sha256_file, strict_json, write_csv, write_json

ENV_ID = "CartPole-v1"
AGENT_COUNT = 2
VARIANT = "single_agent" if AGENT_COUNT == 1 else "two_agents"
TEACHER_GAINS = np.array([0.02, 0.03, 1.0, 0.5], dtype=np.float64)
OBSERVATION_FIELDS = ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]


def teacher_action(observation):
    """A hand-written baseline, not a trained or performance-certified expert."""
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (4,) or not np.isfinite(observation).all():
        raise ValueError("The teacher expects four finite CartPole state values")
    return int(observation @ TEACHER_GAINS > 0)


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return parsed


def validate_seed(seed, episodes=1):
    if not 0 <= seed < 2**32 or seed + episodes - 1 >= 2**32:
        raise ValueError("Seeds must stay within 0 through 2**32 - 1")


def episode_split(episode_ids, validation_fraction=0.2, seed=42):
    """Split whole episodes; neighboring timesteps never straddle the split."""
    episode_ids = np.asarray(episode_ids)
    unique = np.unique(episode_ids)
    if len(unique) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need at least two episodes and a validation fraction in (0, 1)")
    train_ids, validation_ids = train_test_split(unique, test_size=validation_fraction, random_state=seed)
    return np.isin(episode_ids, train_ids), np.isin(episode_ids, validation_ids)


def collect_one(args):
    validate_seed(args.seed, args.episodes)
    if args.episodes < 2 or not 0 <= args.exploration <= 1:
        raise ValueError("Collect at least two episodes; exploration must be in [0, 1]")
    directory = new_directory(args.output_dir)
    rng = np.random.default_rng(args.seed + args.agent_id)
    observations, actions, executed_actions, episode_ids, rows = [], [], [], [], []
    env = gym.make(ENV_ID)
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, _ = env.reset(seed=seed)
            total, length = 0.0, 0
            while True:
                label = teacher_action(observation)
                executed = int(rng.integers(2)) if rng.random() < args.exploration else label
                # Label the CURRENT observation before executing any action.
                observations.append(np.asarray(observation).copy())
                actions.append(label)
                executed_actions.append(executed)
                episode_ids.append(episode)
                observation, reward, terminated, truncated, _ = env.step(executed)
                total += float(reward)
                length += 1
                if terminated or truncated:
                    break
            rows.append({"episode": episode, "seed": seed, "return": total, "length": length,
                         "terminated": bool(terminated), "truncated": bool(truncated)})
    finally:
        env.close()
    with (directory / "demonstrations.npz").open("xb") as handle:
        np.savez_compressed(handle, observations=np.asarray(observations, dtype=np.float32),
                            actions=np.asarray(actions, dtype=np.int64),
                            executed_actions=np.asarray(executed_actions, dtype=np.int64),
                            episode_ids=np.asarray(episode_ids, dtype=np.int64),
                            episode_seeds=np.asarray([r["seed"] for r in rows], dtype=np.int64))
    write_csv(directory / "episodes.csv", rows)
    write_json(directory / "metadata.json", {
        "env_id": ENV_ID, "agent_id": args.agent_id, "agent_count": AGENT_COUNT,
        "teacher": "linear rule baseline", "teacher_gains": TEACHER_GAINS.tolist(),
        "observation_fields": OBSERVATION_FIELDS, "seed": args.seed, "episodes": args.episodes,
        "exploration_probability": args.exploration, "exploration_seed": args.seed + args.agent_id,
        "samples": len(actions),
        "label_semantics": "teacher action for the current pre-step observation",
        "executed_action_semantics": "teacher action or random exploration; never substituted for the label",
        "gymnasium_version": gym.__version__, "numpy_version": np.__version__,
    })
    return {"directory": str(directory), "samples": len(actions), "episodes": len(rows)}


def load_demonstrations(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in ("observations", "actions", "episode_ids", "episode_seeds")}
    x, y, ids = arrays["observations"], arrays["actions"], arrays["episode_ids"]
    if x.ndim != 2 or x.shape[1] != 4 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Observations must have finite shape (N, 4), with N > 0")
    if y.shape != (len(x),) or ids.shape != (len(x),):
        raise ValueError("Actions and episode_ids must each have shape (N,)")
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, [0, 1]).all():
        raise ValueError("Actions must be integer labels 0 or 1")
    if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
        raise ValueError("episode_ids must be nonnegative integers")
    seeds = arrays["episode_seeds"]
    if seeds.ndim != 1 or not len(seeds) or not np.issubdtype(seeds.dtype, np.integer):
        raise ValueError("episode_seeds must be a nonempty integer vector")
    if not np.array_equal(np.unique(ids), np.arange(len(seeds))):
        raise ValueError("episode_ids must cover each episode_seeds entry exactly once or more")
    if (seeds < 0).any() or (seeds >= 2**32).any() or len(np.unique(seeds)) != len(seeds):
        raise ValueError("Collection seeds must be unique and in the supported range")
    return arrays


def train_one(args):
    validate_seed(args.seed)
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    data = load_demonstrations(args.data)
    collection_metadata = strict_json(args.data.parent / "metadata.json")
    if collection_metadata.get("teacher_gains") != TEACHER_GAINS.tolist():
        raise ValueError("Teacher rule changed since collection; collect new demonstrations")
    if collection_metadata.get("agent_id") != args.agent_id or collection_metadata.get("agent_count") != AGENT_COUNT:
        raise ValueError("Demonstration metadata belongs to a different agent or variant")
    train_mask, validation_mask = episode_split(data["episode_ids"], args.validation_fraction, args.seed)
    x, y = data["observations"], data["actions"]
    if set(y[train_mask]) != {0, 1}:
        raise ValueError("Training episodes must include both actions; collect more demonstrations")
    scaler = StandardScaler().fit(x[train_mask])
    x_train, x_validation = scaler.transform(x[train_mask]), scaler.transform(x[validation_mask])
    classifier = MLPClassifier(hidden_layer_sizes=(32, 32), activation="relu", solver="adam",
                               learning_rate_init=args.learning_rate,
                               batch_size=min(args.batch_size, int(train_mask.sum())),
                               random_state=args.seed, shuffle=True)
    directory = new_directory(args.output_dir)
    history, best_score, best_classifier, best_epoch = [], -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        # Each partial_fit call performs one epoch of supervised optimization.
        classifier.partial_fit(x_train, y[train_mask], classes=np.array([0, 1]))
        training_accuracy = float(accuracy_score(y[train_mask], classifier.predict(x_train)))
        validation_accuracy = float(accuracy_score(y[validation_mask], classifier.predict(x_validation)))
        history.append({"epoch": epoch, "loss": float(classifier.loss_),
                        "train_accuracy": training_accuracy, "validation_accuracy": validation_accuracy})
        if validation_accuracy > best_score:
            best_score, best_epoch = validation_accuracy, epoch
            best_classifier = copy.deepcopy(classifier)
    model = Pipeline([("scaler", scaler), ("classifier", best_classifier)])
    metadata = {
        "format_version": 1, "env_id": ENV_ID, "seed": args.seed,
        "agent_id": args.agent_id, "agent_count": AGENT_COUNT,
        "teacher_gains": TEACHER_GAINS.tolist(), "observation_fields": OBSERVATION_FIELDS,
        "data_sha256": sha256_file(args.data), "collection_seeds": data["episode_seeds"].tolist(),
        "training_episode_ids": np.unique(data["episode_ids"][train_mask]).tolist(),
        "validation_episode_ids": np.unique(data["episode_ids"][validation_mask]).tolist(),
        "best_epoch": best_epoch, "best_validation_accuracy": best_score,
        "selection_metric": "validation action accuracy, not online return",
        "parameters": {"epochs": args.epochs, "batch_size": args.batch_size,
                       "learning_rate": args.learning_rate, "validation_fraction": args.validation_fraction,
                       "hidden_layer_sizes": [32, 32]},
        "versions": {"gymnasium": gym.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__},
    }
    with (directory / "student.joblib").open("xb") as handle:
        joblib.dump({"model": model, "metadata": metadata}, handle)
    write_json(directory / "metadata.json", metadata)
    write_csv(directory / "training.csv", history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        axes[0].plot([r["epoch"] for r in history], [r["loss"] for r in history])
        axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Behavior-cloning training loss")
        for key in ("train_accuracy", "validation_accuracy"):
            axes[1].plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        axes[1].set(xlabel="Epoch", ylabel="Action accuracy", ylim=(0, 1.01))
        axes[1].legend()
        fig.tight_layout()
        with (directory / "training.png").open("xb") as handle:
            fig.savefig(handle, format="png", dpi=150)
    finally:
        plt.close(fig)
    return {"directory": str(directory), "best_epoch": best_epoch, "validation_action_accuracy": best_score}


def load_student(path):
    # Only load model files you generated yourself: joblib uses pickle internally.
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload or "metadata" not in payload:
        raise ValueError("Invalid student artifact")
    metadata = payload["metadata"]
    if metadata.get("format_version") != 1 or metadata.get("env_id") != ENV_ID:
        raise ValueError("Unsupported student artifact")
    return payload


def evaluate_one(args):
    validate_seed(args.seed, args.episodes)
    saved = load_student(args.model)
    if saved["metadata"].get("agent_id") != args.agent_id or saved["metadata"].get("agent_count") != AGENT_COUNT:
        raise ValueError("Student model belongs to a different agent or variant")
    if saved["metadata"]["teacher_gains"] != TEACHER_GAINS.tolist():
        raise ValueError("Teacher rule changed since model training; use the matching source version")
    evaluation_seeds = set(range(args.seed, args.seed + args.episodes))
    if evaluation_seeds & set(saved["metadata"]["collection_seeds"]):
        raise ValueError("Evaluation seeds overlap collection seeds; choose fresh seeds")
    directory = new_directory(args.output_dir)
    rows = []
    for policy in ("teacher", "student", "random"):
        env = gym.make(ENV_ID)
        try:
            for episode in range(args.episodes):
                seed = args.seed + episode
                observation, _ = env.reset(seed=seed)
                rng = np.random.default_rng(seed)
                total, length = 0.0, 0
                while True:
                    if policy == "teacher":
                        action = teacher_action(observation)
                    elif policy == "student":
                        action = int(saved["model"].predict(np.asarray(observation).reshape(1, 4))[0])
                    else:
                        action = int(rng.integers(2))
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    length += 1
                    if terminated or truncated:
                        break
                rows.append({"policy": policy, "episode": episode, "seed": seed, "return": total,
                             "length": length, "terminated": bool(terminated), "truncated": bool(truncated)})
        finally:
            env.close()
    summary = []
    for policy in ("teacher", "student", "random"):
        values = np.array([r["return"] for r in rows if r["policy"] == policy])
        summary.append({"policy": policy, "episodes": len(values), "mean_return": float(values.mean()),
                        "std_return": float(values.std(ddof=1)) if len(values) > 1 else None})
    write_csv(directory / "episodes.csv", rows)
    write_json(directory / "summary.json", {"env_id": ENV_ID, "seed": args.seed, "episodes_per_policy": args.episodes,
               "model_sha256": sha256_file(args.model), "matched_initial_seeds": True, "summary": summary})
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        ax.bar([r["policy"] for r in summary], [r["mean_return"] for r in summary],
               yerr=[r["std_return"] or 0 for r in summary], capsize=5)
        ax.set(ylabel="Episode return", title="Fresh matched seeds: mean return +/- sample SD")
        fig.tight_layout()
        with (directory / "comparison.png").open("xb") as handle:
            fig.savefig(handle, format="png", dpi=150)
    finally:
        plt.close(fig)
    return {"directory": str(directory), "summary": summary}


def agent_arguments(args, agent_id, **updates):
    return SimpleNamespace(**{**vars(args), "agent_id": agent_id, **updates})


def collect(args):
    validate_seed(args.seed, args.episodes)
    validate_seed(args.seed, AGENT_COUNT)
    if args.episodes < 2 or not 0 <= args.exploration <= 1:
        raise ValueError("Collect at least two episodes; exploration must be in [0, 1]")
    directory = new_directory(args.output_dir)
    results = []
    for agent_id in range(AGENT_COUNT):
        # Each call creates and closes its own CartPole environment.
        result = collect_one(agent_arguments(args, agent_id, output_dir=directory / f"agent_{agent_id}"))
        results.append({"agent_id": agent_id, **result})
    write_json(directory / "agents.json", {"variant": VARIANT, "agents": AGENT_COUNT,
               "episodes_per_agent": args.episodes, "total_episodes": args.episodes * AGENT_COUNT,
               "execution": "sequential independent environments; no interaction or shared actions", "results": results})
    return {"directory": str(directory), "agents": AGENT_COUNT, "results": results}


def train(args):
    validate_seed(args.seed, AGENT_COUNT)
    for agent_id in range(AGENT_COUNT):
        load_demonstrations(args.data_dir / f"agent_{agent_id}" / "demonstrations.npz")
    directory = new_directory(args.output_dir)
    results = []
    for agent_id in range(AGENT_COUNT):
        # Fresh scaler, weights and optimizer state for every learner.
        result = train_one(agent_arguments(args, agent_id, seed=args.seed + agent_id,
                           data=args.data_dir / f"agent_{agent_id}" / "demonstrations.npz",
                           output_dir=directory / f"agent_{agent_id}"))
        results.append({"agent_id": agent_id, **result})
    write_json(directory / "agents.json", {"variant": VARIANT, "agents": AGENT_COUNT,
               "independent_models": True, "epochs_per_agent": args.epochs, "results": results})
    return {"directory": str(directory), "agents": AGENT_COUNT, "results": results}


def evaluate(args):
    validate_seed(args.seed, args.episodes)
    for agent_id in range(AGENT_COUNT):
        saved = load_student(args.model_dir / f"agent_{agent_id}" / "student.joblib")
        if set(range(args.seed, args.seed + args.episodes)) & set(saved["metadata"]["collection_seeds"]):
            raise ValueError("Evaluation seeds overlap collection seeds; choose fresh seeds")
    directory = new_directory(args.output_dir)
    rows, results = [], []
    for agent_id in range(AGENT_COUNT):
        result = evaluate_one(agent_arguments(args, agent_id,
                              model=args.model_dir / f"agent_{agent_id}" / "student.joblib",
                              output_dir=directory / f"agent_{agent_id}"))
        results.append({"agent_id": agent_id, **result})
        with (directory / f"agent_{agent_id}" / "episodes.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"agent_id": agent_id, "policy": row["policy"], "seed": int(row["seed"]),
                             "return": float(row["return"]), "length": int(row["length"])})
    summaries = []
    for policy in ("teacher", "student", "random"):
        # Agents sharing a seed are correlated. Average them BEFORE computing SD across seeds.
        per_seed = [np.mean([r["return"] for r in rows if r["policy"] == policy and r["seed"] == seed])
                    for seed in range(args.seed, args.seed + args.episodes)]
        summaries.append({"policy": policy, "agents": AGENT_COUNT, "evaluation_seeds": args.episodes,
                          "mean_return_per_agent": float(np.mean(per_seed)),
                          "std_across_seed_means": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None})
    write_csv(directory / "all_agents.csv", rows)
    write_json(directory / "summary.json", {"variant": VARIANT, "agents": AGENT_COUNT, "results": results,
               "matched_initial_seeds": True, "aggregation": "mean across agents per seed, then mean/SD across seeds",
               "summary": summaries})
    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        ax.bar([r["policy"] for r in summaries], [r["mean_return_per_agent"] for r in summaries],
               yerr=[r["std_across_seed_means"] or 0 for r in summaries], capsize=5)
        ax.set(ylabel="Mean return per agent", title=f"{AGENT_COUNT} independent agent(s): mean +/- SD across seeds")
        fig.tight_layout()
        with (directory / "comparison.png").open("xb") as handle:
            fig.savefig(handle, format="png", dpi=150)
    finally:
        plt.close(fig)
    return {"directory": str(directory), "agents": AGENT_COUNT, "summary": summaries}


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("collect", help="collect teacher-labeled state/action demonstrations")
    command.add_argument("--output-dir", type=Path, default=Path("runs") / VARIANT / "demos")
    command.add_argument("--episodes", type=positive, default=50)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--exploration", type=float, default=0.05)
    command.set_defaults(handler=collect)

    command = commands.add_parser("train", help="train an MLP to copy the teacher's action labels")
    command.add_argument("--data-dir", type=Path, default=Path("runs") / VARIANT / "demos")
    command.add_argument("--output-dir", type=Path, default=Path("runs") / VARIANT / "bc")
    command.add_argument("--epochs", type=positive, default=50)
    command.add_argument("--batch-size", type=positive, default=64)
    command.add_argument("--learning-rate", type=float, default=0.001)
    command.add_argument("--validation-fraction", type=float, default=0.2)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(handler=train)

    command = commands.add_parser("evaluate", help="compare teacher, student and random actions on fresh matched seeds")
    command.add_argument("--model-dir", type=Path, default=Path("runs") / VARIANT / "bc")
    command.add_argument("--output-dir", type=Path, default=Path("runs") / VARIANT / "evaluation")
    command.add_argument("--episodes", type=positive, default=20)
    command.add_argument("--seed", type=int, default=10000)
    command.set_defaults(handler=evaluate)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
