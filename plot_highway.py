"""Plot completed Highway evaluation episodes from CSV or existing evaluation JSON."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from practice_io import new_directory, sha256_file, strict_json, write_csv, write_json

REQUIRED = ("seed", "return", "length", "crashed", "mean_speed_mps")


def boolean(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError(f"{name} must be true/false or 0/1, got {value!r}")


def finite(value, name, minimum=None, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"Invalid finite value for {name}: {value!r}")
    if integer and not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(result) if integer else result


def normalize(row, *, default_label, default_step=None, config_id="unknown", synthetic=False):
    if not isinstance(row, dict):
        raise ValueError("Each episode must be an object")
    row = dict(row)
    if "length" not in row and "steps" in row:
        row["length"] = row["steps"]
    missing = [name for name in REQUIRED if name not in row or row[name] is None or row[name] == ""]
    if missing:
        raise ValueError(f"Missing episode fields: {', '.join(missing)}")
    policy = row.get("policy", row.get("method", default_label))
    if not isinstance(policy, str) or not policy.strip() or any(ord(c) < 32 for c in policy):
        raise ValueError("policy must be a nonempty label without control characters")
    step = row.get("checkpoint_step", row.get("global_step", default_step))
    step = None if step in (None, "") else finite(step, "checkpoint_step", minimum=0, integer=True)
    return {
        "policy": policy, "seed": finite(row["seed"], "seed", minimum=0, integer=True),
        "checkpoint_step": step, "return": finite(row["return"], "return"),
        "length": finite(row["length"], "length", minimum=1, integer=True),
        "crashed": boolean(row["crashed"], "crashed"),
        "mean_speed_mps": finite(row["mean_speed_mps"], "mean_speed_mps", minimum=0),
        "config_id": str(row.get("config_id", config_id)),
        "data_kind": "synthetic" if synthetic or row.get("data_kind") == "synthetic" else "user-supplied",
    }


def read_evaluation(path, label=None):
    path = Path(path)
    metadata = {}
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            episodes = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        raw = strict_json(path)
        if isinstance(raw, list):
            episodes = raw
        elif isinstance(raw, dict) and isinstance(raw.get("episodes"), list):
            episodes, metadata = raw["episodes"], raw
        elif isinstance(raw, dict) and isinstance(raw.get("summary"), dict):
            episodes, metadata = [raw["summary"]], raw
        else:
            raise ValueError("JSON must be an episode array, an episodes object, or a summary/trace object")
    else:
        raise ValueError("Inputs must be CSV or JSON evaluation files")
    default_label = label or Path(str(metadata.get("checkpoint", path.stem))).stem
    config = metadata.get("env_config")
    config_id = "unknown" if config is None else hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    rows, excluded = [], 0
    for index, episode in enumerate(episodes):
        try:
            if isinstance(episode, dict) and boolean(episode.get("runner_cutoff", False), "runner_cutoff"):
                excluded += 1
                continue
            if label is not None:
                episode = {**episode, "policy": label}
            rows.append(normalize(episode, default_label=default_label,
                                  default_step=metadata.get("global_step"), config_id=config_id,
                                  synthetic=metadata.get("data_kind") == "synthetic"))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path.name}, episode {index}: {exc}") from exc
    return rows, excluded


def validate_rows(rows):
    if not rows:
        raise ValueError("No completed evaluation episodes remain")
    seen = set()
    for row in rows:
        key = (row["policy"], row["checkpoint_step"], row["seed"])
        if key in seen:
            raise ValueError(f"Duplicate policy/checkpoint/seed: {key}; do not count the same episode twice")
        seen.add(key)
    known_configs = {row["config_id"] for row in rows} - {"unknown"}
    if len(known_configs) > 1:
        raise ValueError("Different known environment configurations were supplied; plot each configuration separately")


def wilson_interval(successes, count, z=1.959963984540054):
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("Wilson interval needs 0 <= successes <= a positive count")
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def group_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["policy"], row["checkpoint_step"]), []).append(row)
    return sorted(grouped.items(), key=lambda pair: (pair[0][0], -1 if pair[0][1] is None else pair[0][1]))


def summarize(rows):
    summaries = []
    for (policy, step), records in group_rows(rows):
        result = {"policy": policy, "checkpoint_step": step, "episodes": len(records)}
        for name in ("return", "length", "mean_speed_mps"):
            values = np.asarray([r[name] for r in records], dtype=float)
            result[name + "_mean"] = float(values.mean())
            result[name + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        crashed = sum(r["crashed"] for r in records)
        low, high = wilson_interval(crashed, len(records))
        result.update(crash_count=crashed, crash_rate=crashed / len(records),
                      crash_ci95_low=low, crash_ci95_high=high)
        summaries.append(result)
    return summaries


def label_for(policy, step):
    return policy if step is None else f"{policy}\nstep={step}"


def save_figure(fig, path, synthetic):
    try:
        if synthetic:
            fig.suptitle("SYNTHETIC EXAMPLE DATA - NOT MEASURED PERFORMANCE", fontsize=10, color="darkred")
        fig.tight_layout(rect=(0, 0, 1, 0.95) if synthetic else (0, 0, 1, 1))
        with Path(path).open("xb") as handle:
            fig.savefig(handle, format="png", dpi=160)
    finally:
        plt.close(fig)


def create_plots(rows, summaries, directory):
    synthetic = any(row["data_kind"] == "synthetic" for row in rows)
    labels = [label_for(row["policy"], row["checkpoint_step"]) for row in summaries]
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(labels) * 1.1), 8))
    panels = [("return", "Episode return"), ("mean_speed_mps", "Mean speed [m/s]"), ("length", "Episode length [steps]")]
    for ax, (key, title) in zip([axes[0, 0], axes[1, 0], axes[1, 1]], panels):
        ax.bar(labels, [s[key + "_mean"] for s in summaries],
               yerr=[s[key + "_std"] or 0 for s in summaries], capsize=4)
        ax.set(title=title + " (mean +/- sample SD)", ylabel=title)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
    ax = axes[0, 1]
    rates = np.array([s["crash_rate"] for s in summaries]) * 100
    errors = np.array([[s["crash_rate"] - s["crash_ci95_low"] for s in summaries],
                       [s["crash_ci95_high"] - s["crash_rate"] for s in summaries]]) * 100
    ax.bar(labels, rates, yerr=np.maximum(errors, 0), capsize=4, color="tab:orange")
    ax.set(title="Crash rate (95% Wilson interval)", ylabel="Crashed episodes [%]", ylim=(0, 100))
    ax.tick_params(axis="x", labelrotation=25)
    save_figure(fig, directory / "overview.png", synthetic)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (policy, step), records in group_rows(rows):
        ax.hist([r["return"] for r in records], bins="auto", alpha=0.45,
                label=f"{label_for(policy, step).replace(chr(10), ' ')} (n={len(records)})")
    ax.set(title="Return distributions across evaluation episodes", xlabel="Episode return", ylabel="Episodes")
    ax.legend(fontsize=8)
    save_figure(fig, directory / "return_distribution.png", synthetic)

    curves = {}
    for row in summaries:
        if row["checkpoint_step"] is not None:
            curves.setdefault(row["policy"], []).append(row)
    curves = {policy: records for policy, records in curves.items() if len(records) >= 2}
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for policy, records in curves.items():
            records.sort(key=lambda row: row["checkpoint_step"])
            steps = np.array([r["checkpoint_step"] for r in records])
            for ax, (key, title) in zip([axes[0], axes[2]], panels[:2]):
                means = np.array([r[key + "_mean"] for r in records])
                std = np.array([r[key + "_std"] or 0 for r in records])
                ax.plot(steps, means, marker="o", label=policy)
                ax.fill_between(steps, means - std, means + std, alpha=0.15)
                ax.set(xlabel="Checkpoint training step", ylabel=title, title="Mean +/- episode sample SD")
            axes[1].plot(steps, [100 * r["crash_rate"] for r in records], marker="o", label=policy)
        axes[1].set(xlabel="Checkpoint training step", ylabel="Crashed episodes [%]", ylim=(0, 100))
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        save_figure(fig, directory / "checkpoint_curves.png", synthetic)


def run(args):
    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("Supply one --labels entry for each input file")
    rows, excluded, inputs = [], 0, []
    for index, path in enumerate(args.inputs):
        loaded, skipped = read_evaluation(path, args.labels[index] if args.labels else None)
        rows.extend(loaded)
        excluded += skipped
        inputs.append({"file": str(path), "sha256": sha256_file(path), "episodes": len(loaded), "excluded_cutoffs": skipped})
    validate_rows(rows)
    summaries = summarize(rows)
    directory = new_directory(args.output_dir)
    write_csv(directory / "normalized_episodes.csv", rows)
    write_csv(directory / "summary.csv", summaries)
    write_json(directory / "report.json", {
        "inputs": inputs, "included_episodes": len(rows), "excluded_runner_cutoffs": excluded,
        "synthetic_example": any(row["data_kind"] == "synthetic" for row in rows),
        "uncertainty": {"continuous_metrics": "sample SD across evaluation episodes; not a confidence interval",
                        "crash_rate": "95% Wilson interval; assumes independent episodes"},
        "comparability": "Descriptive plots only. Check environment settings, seeds and observation access before ranking policies.",
        "seed_sets": [{"policy": policy, "checkpoint_step": step, "seeds": sorted(r["seed"] for r in records)}
                      for (policy, step), records in group_rows(rows)],
        "summary": summaries,
    })
    create_plots(rows, summaries, directory)
    return {"output_dir": str(directory), "episodes": len(rows), "excluded_cutoffs": excluded}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", help="one policy label per input; overrides labels inside those files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/highway_plots"))
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

