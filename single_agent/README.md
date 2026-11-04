# 1 independent CartPole agent(s)

This variant has `AGENT_COUNT = 1`. Each learner has its own environment, demonstration data, scaler, MLP weights, optimizer state, and output directory. Agents run sequentially and do not interact.

From the repository root, after installing its dependencies:

```bash
python -m single_agent.cartpole_bc collect --episodes 50 --seed 0
python -m single_agent.cartpole_bc train --epochs 50 --seed 42
python -m single_agent.cartpole_bc evaluate --episodes 20 --seed 10000
```

Use `--help` after each subcommand for paths and hyperparameters. Default outputs are under `runs/single_agent/`.

- Collection: 50 episodes **per agent**, 50 episodes total at the defaults.
- Training: 50 epochs per agent, using only that agent's demonstrations.
- Environments use the same episode-seed list across agents; exploration RNGs use `seed + agent_id`.
- Models and episode splits use `train_seed + agent_id`.
- Evaluation uses fresh seeds, identical initial seeds for teacher/student/random, and separate per-agent records.
- The root evaluation summary reports return per agent. It averages agents within each seed before taking mean/SD across seeds.

Teacher action accuracy is not a guarantee of balancing performance. The rule teacher and all results remain unmeasured: collection, training, evaluation and tests were not run during creation.

See the [root README](../README.md) for labeling semantics, output descriptions, dependencies and the one-versus-two comparison. Every corresponding file is listed in the [diff index](../diffs/README.md).
