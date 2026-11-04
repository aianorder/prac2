# prac2

Two small practice projects in one repository:

1. **CartPole imitation learning** with one agent or two independent agents.
2. **Highway evaluation visualization** from saved CSV/JSON episode results.

**Execution status:** code and documentation only. Installation, demonstration collection, learning, environment evaluation, plotting, and pytest were intentionally not run when this repository was created. No trained models, measured performance, or generated evaluation figures are included. The source-only diff generator was run to produce the requested patch files.

## Setup

Use Python 3.11 or newer; Python 3.12 is a convenient starting point.

```bash
git clone https://github.com/aianorder/prac2.git
cd prac2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, use `python` instead of `python3` and activate with `.venv\Scripts\Activate.ps1` in PowerShell. Use `python -m pip install -e .` if you do not need pytest.

## Project layout

```text
single_agent/          One learner controlling its own CartPole
two_agents/            Two separate learners, each controlling its own CartPole
diffs/                 Every corresponding file's diff, hashes, and combined patch
plot_highway.py        CSV/JSON -> performance plots and statistical summaries
practice_io.py         Shared output helpers
make_diffs.py          Recreate source-file diffs without running the programs
example_highway.csv   Explicitly synthetic plotting data
HIGHWAY_FORMAT.md      Input schemas, compatible JSON formats and graph interpretation
test_highway_plot.py  Optional pytest source
pyproject.toml         Package dependencies and command-line entry points
```

## CartPole: one agent

```bash
# 1. Record teacher labels for the current state.
python -m single_agent.cartpole_bc collect --episodes 50 --seed 0

# 2. Fit a student MLP using supervised classification.
python -m single_agent.cartpole_bc train --epochs 50 --seed 42

# 3. Compare student, teacher and random actions on fresh matched seeds.
python -m single_agent.cartpole_bc evaluate --episodes 20 --seed 10000
```

Outputs go to `runs/single_agent/`. The saved model is `runs/single_agent/bc/agent_0/student.joblib`.

The installed entry point is `prac-cartpole-single`. See [the one-agent guide](single_agent/README.md).

## CartPole: two independent agents

```bash
python -m two_agents.cartpole_bc collect --episodes 50 --seed 0
python -m two_agents.cartpole_bc train --epochs 50 --seed 42
python -m two_agents.cartpole_bc evaluate --episodes 20 --seed 10000
```

Outputs go to `runs/two_agents/`, with separate `agent_0/` and `agent_1/` directories for demonstrations, scalers, classifier weights, training history, and evaluation records.

The installed entry point is `prac-cartpole-two`. See [the two-agent guide](two_agents/README.md).

| Contract | One agent | Two agents |
| --- | --- | --- |
| CartPole environments | One independent environment | One independent environment per agent |
| Learned models | One | Two; no shared weights or optimizer state |
| Collection budget at `--episodes 50` | 50 episodes | 50 per agent, 100 total |
| Training budget at `--epochs 50` | 50 epochs | 50 per agent |
| Collection environment seeds | `seed + episode` | Same seed list for each agent |
| Exploration RNG | `seed` | `seed + agent_id` |
| Model/split RNG | `train_seed` | `train_seed + agent_id` |
| Online evaluation | Teacher/student/random on matched fresh seeds | Same comparison for each agent on the same fresh seed list |
| Execution | Sequential | Sequential independent runs; no multiprocessing |

Two agents here means **two independent CartPole control problems**, as opposed to joint control of one cart. There is no communication, shared reward, collision between carts, or multi-agent physics modification. Running two agents is not advertised as a parallel speedup.

For a one-versus-two comparison, use **return per agent**, not the sum of two agents' returns. The two-agent aggregate first averages agents within each evaluation seed, then computes mean and sample SD across seeds. This avoids treating agents evaluated on the same seeds as extra independent evaluation worlds.

## How behavior cloning works

CartPole observations contain cart position, cart velocity, pole angle, and pole angular velocity. Its two actions push left or right. The environment is Gymnasium's standard [`CartPole-v1`](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

The hand-written teacher uses:

```text
score = 0.02 * cart_position + 0.03 * cart_velocity
        + 1.00 * pole_angle + 0.50 * pole_angular_velocity
teacher_action = 1 if score > 0 else 0
```

This is an educational rule baseline, not a trained or performance-certified expert. Its actual balancing ability has not been measured here.

During collection, the code labels the **current, pre-step observation** with the teacher's action. With probability `--exploration` (default 0.05), a random action is executed to visit additional states. The executed action is recorded separately; the supervised label remains the teacher's action at that same state.

Training fits a `StandardScaler` and a `(32, 32)` ReLU `MLPClassifier`. Its loss is cross-entropy on teacher action labels. No reward is used to train the student. Whole episodes are split into training/validation sets so neighboring steps do not leak between them. The best checkpoint is selected by validation action accuracy.

**Action accuracy is not online control performance.** A small error can move the student into states absent from demonstrations. The separate online evaluation compares returns on seeds that were not used for collection and rejects overlap.

## CartPole output files

Each agent gets its own subdirectory:

- `demos/agent_N/demonstrations.npz`: observations, teacher labels, executed actions, episode IDs and collection seeds.
- `demos/agent_N/episodes.csv` and `metadata.json`: collection returns and provenance.
- `bc/agent_N/student.joblib`: fitted scaler and student, with training metadata.
- `bc/agent_N/training.csv` and `training.png`: loss and action accuracy by epoch.
- `evaluation/agent_N/episodes.csv`, `summary.json`, `comparison.png`: teacher/student/random results.
- `evaluation/all_agents.csv`, `summary.json`, `comparison.png`: cross-agent records and return-per-agent aggregates.

All commands protect existing nonempty output directories. Use a new directory for another experiment. Only load your own generated `.joblib` models; joblib uses pickle internally.

## Every file's one-agent/two-agent diff

Start at [`diffs/README.md`](diffs/README.md). It links to a separate patch for **every corresponding variant file** and records shared root files as unchanged. [`diffs/manifest.json`](diffs/manifest.json) contains status and both SHA256 hashes; [`diffs/all_changes.patch`](diffs/all_changes.patch) combines the changed-file hunks.

The control/training code intentionally differs only in the `AGENT_COUNT` constant. Both variants contain a full copy of the same implementation for direct source comparison. Their guides, package descriptions and variant expectations in tests also have individual diffs. Shared utilities and Highway plotting do not change with the CartPole agent count.

Regenerate the diff files, without running any learning code:

```bash
python make_diffs.py
```

## Highway evaluation graphs

Plot existing PPO and rule-controller evaluation files directly:

```bash
python plot_highway.py path/to/model.evaluation.json path/to/summary.json \
  --labels PPO Rule --output-dir outputs/highway-comparison
```

The plotter supports the existing PPO `episodes` JSON format, rule-based `summary.json`, single-episode `summary`/`trace` JSON, and a simple CSV format. It draws:

- Mean episode return with sample standard deviation.
- Crash rate with a 95% Wilson interval.
- Mean speed and episode length.
- Return distributions.
- Checkpoint performance curves when checkpoint training steps are provided.

It writes normalized episode records, summaries and provenance alongside the figures. Incomplete runner cutoffs are excluded. Duplicate policy/checkpoint/seed entries are rejected. See [HIGHWAY_FORMAT.md](HIGHWAY_FORMAT.md) for the exact schema and statistical limits.

The included CSV is **synthetic example data**, not measured agent performance:

```bash
python plot_highway.py example_highway.csv --output-dir outputs/highway-example
```

These commands are instructions for later use; they were not executed during creation.

## Optional tests

```bash
python -m pytest -q
```

Test files cover labeling, episode-level splitting, independent per-agent output/model settings, input validation, existing Highway JSON compatibility, duplicate detection, and statistical summaries. They are included as source only and were not executed. No automatic training, evaluation, or CI workflow is configured.

## References

- [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [HighwayEnv](https://highway-env.farama.org/quickstart/)
- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

The repository contains reusable source, explanatory documentation, patches, and explicitly synthetic example data. Demonstrations, trained models, private evaluation logs, generated images, virtual environments, and credentials stay outside Git.
