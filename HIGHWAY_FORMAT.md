# Highway evaluation plots

`plot_highway.py` reads existing evaluation results. It does not run Highway, train a driving policy, or load a checkpoint. No `highway-env` installation is needed for plotting saved results.

## Supported inputs

### 1. Episode CSV

```csv
policy,seed,return,length,crashed,mean_speed_mps
PPO,10000,30.2,40,false,25.1
PPO,10001,18.5,24,true,23.0
```

Required episode fields: `seed`, `return`, `length`, `crashed`, `mean_speed_mps`.

- `policy` is the comparison label. If omitted, the input filename supplies the label.
- `steps` is accepted as an alias for `length`.
- `method` is accepted as an alias for `policy`.
- `checkpoint_step` (or `global_step`) optionally records the training step of a checkpoint.
- `crashed` must be a JSON boolean, `true`/`false`, or `0`/`1`. The string `False` is parsed as false.
- `runner_cutoff=true` excludes an unfinished episode. A cutoff is not a successful completed episode.
- Optional `config_id` identifies an evaluation condition. Different known configuration IDs cannot be silently pooled.
- `data_kind=synthetic` marks illustrative data and adds a synthetic-data notice to the plots.

### 2. PPO evaluation JSON

The existing `checkpoint`, `global_step`, `episodes`, `summary` format is supported directly:

```json
{
  "checkpoint": "policy.pt",
  "global_step": 153600,
  "action_selection": "argmax",
  "episodes": [
    {"seed": 10000, "return": 30.2, "length": 40, "crashed": false,
     "mean_speed_mps": 25.1, "terminated": false, "truncated": true}
  ]
}
```

The top-level `summary` is ignored: statistics are recomputed from episode records. Top-level `global_step` becomes the checkpoint step for every episode.

### 3. Rule-based evaluation JSON

Both `summary.json` with an `episodes` array and a single-episode `{"summary": {...}, "trace": [...]}` file are supported. `steps` is translated to `length`. The trace is not needed for performance comparison.

If a JSON document includes `env_config`, its sorted contents are hashed into a configuration ID. If configuration provenance is absent, it is recorded as unknown, not assumed equal to another file's configuration.

## Usage

```bash
# Existing PPO and rule-controller evaluation outputs:
python plot_highway.py path/to/model.evaluation.json path/to/summary.json \
  --labels PPO Rule --output-dir outputs/highway-comparison

# Repeated checkpoints from the same policy:
python plot_highway.py path/to/step10000.json path/to/step20000.json \
  --labels PPO PPO --output-dir outputs/highway-checkpoints

# An illustrative example only; these values are fabricated:
python plot_highway.py example_highway.csv --output-dir outputs/highway-example
```

Supply exactly one label for each input if using `--labels`. Explicit labels override policy names inside those files. Without the option, CSV policy columns are preserved. Duplicate `(policy, checkpoint_step, seed)` records are errors so that an episode cannot be accidentally counted twice.

## Outputs

| File | Contents |
| --- | --- |
| `overview.png` | Mean return, crash rate, mean speed, episode length |
| `return_distribution.png` | Return distributions for each policy/checkpoint |
| `checkpoint_curves.png` | Return, crash-rate and speed curves when at least two steps exist for a policy |
| `normalized_episodes.csv` | Parsed episode records, preserving seeds and condition IDs |
| `summary.csv` | Grouped counts, means, standard deviations and crash intervals |
| `report.json` | Input hashes, excluded cutoff counts, seed sets, interpretation notes and summaries |

Use a new or empty output directory. Source files and previous reports are preserved.

## Interpretation

Continuous error bars/bands show **sample standard deviation across evaluation episodes**, not a confidence interval and not variation across independently trained agents. For one episode, SD is unknown (`null` in JSON, blank in CSV); the plot omits its error bar.

Crash error bars show **95% Wilson intervals** for the proportion of crashed episodes. Their interpretation assumes independent evaluation episodes. A zero observed crash rate does not establish zero underlying crash probability.

The plots are descriptive. Before ranking policies, match environment configuration, seed sets, episode horizon, action selection, and access to observations. In particular, a rule controller using simulator internals may have more information than a neural policy receiving only standard observations. The plotter does not make those conditions equivalent.

The included `example_highway.csv` is entirely synthetic. It is not an evaluation result for any trained agent.

