"""Optional pytest tests. Not executed when this repository was created."""

import json

import numpy as np
import pytest

from plot_highway import normalize, read_evaluation, summarize, validate_rows, wilson_interval


def episode(**updates):
    return {"seed": 10000, "return": 30, "length": 40, "crashed": False,
            "mean_speed_mps": 25, **updates}


def test_false_string_is_not_a_crash():
    result = normalize(episode(crashed="False"), default_label="policy")
    assert result["crashed"] is False


@pytest.mark.parametrize("updates", [{"return": "NaN"}, {"crashed": "sometimes"},
                                      {"length": 0}, {"mean_speed_mps": -1}, {"seed": 1.5}])
def test_invalid_metrics_are_rejected(updates):
    with pytest.raises(ValueError):
        normalize(episode(**updates), default_label="policy")


def test_existing_ppo_evaluation_json(tmp_path):
    path = tmp_path / "model.evaluation.json"
    path.write_text(json.dumps({"checkpoint": "model.pt", "global_step": 153600,
                                "episodes": [episode()]}))
    rows, excluded = read_evaluation(path, "PPO")
    assert excluded == 0
    assert rows[0]["policy"] == "PPO"
    assert rows[0]["checkpoint_step"] == 153600


def test_existing_rule_summary_steps_alias_and_cutoffs(tmp_path):
    path = tmp_path / "summary.json"
    row = episode()
    row["steps"] = row.pop("length")
    path.write_text(json.dumps({"episodes": [row, episode(seed=10001, runner_cutoff=True)], "env_config": {"duration": 40}}))
    rows, excluded = read_evaluation(path, "Rule")
    assert rows[0]["length"] == 40
    assert len(rows) == excluded == 1


def test_duplicates_do_not_inflate_episode_count():
    row = normalize(episode(), default_label="policy")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows([row, row])


def test_summary_and_wilson_interval():
    rows = [normalize(episode(seed=0, **{"return": 10}), default_label="policy"),
            normalize(episode(seed=1, crashed=True, **{"return": 20}), default_label="policy")]
    summary = summarize(rows)[0]
    assert summary["return_mean"] == 15
    assert np.isclose(summary["return_std"], np.std([10, 20], ddof=1))
    assert summary["crash_rate"] == 0.5
    assert summary["crash_ci95_low"] < 0.5 < summary["crash_ci95_high"]
    low, high = wilson_interval(0, 10)
    assert low >= 0 and high > 0


def test_single_episode_sd_is_unknown():
    assert summarize([normalize(episode(), default_label="p")])[0]["return_std"] is None
