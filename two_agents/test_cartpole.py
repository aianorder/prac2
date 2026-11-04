"""Optional test source; not executed during repository creation."""

from types import SimpleNamespace

import numpy as np
import pytest

from . import cartpole_bc as bc

EXPECTED_AGENTS = 2


def test_variant_agent_count():
    assert bc.AGENT_COUNT == EXPECTED_AGENTS


def test_teacher_action_labels_and_validation():
    assert bc.teacher_action([0, 0, 0.1, 0]) == 1
    assert bc.teacher_action([0, 0, -0.1, 0]) == 0
    with pytest.raises(ValueError):
        bc.teacher_action([0, 0, float("nan"), 0])


def test_whole_episode_split():
    ids = np.repeat(np.arange(10), 5)
    train, validation = bc.episode_split(ids, seed=42)
    assert not set(ids[train]) & set(ids[validation])
    assert np.all(train | validation)
    assert not np.any(train & validation)


def test_missing_second_episode_is_rejected():
    with pytest.raises(ValueError):
        bc.episode_split(np.zeros(10, dtype=int))


def test_collection_uses_separate_output_directories(monkeypatch, tmp_path):
    captured = []

    def fake_collect(args):
        captured.append(args)
        return dict(directory=str(args.output_dir), samples=10, episodes=args.episodes)

    monkeypatch.setattr(bc, "collect_one", fake_collect)
    args = SimpleNamespace(seed=0, episodes=3, exploration=0.05, output_dir=tmp_path / "demos")
    result = bc.collect(args)
    assert result["agents"] == EXPECTED_AGENTS
    assert len(captured) == EXPECTED_AGENTS
    assert len(set(a.output_dir for a in captured)) == EXPECTED_AGENTS
    assert [a.agent_id for a in captured] == list(range(EXPECTED_AGENTS))
    assert all(a.seed == 0 for a in captured)


def test_training_uses_independent_model_seeds_and_data_paths(monkeypatch, tmp_path):
    captured = []

    def fake_train(args):
        captured.append(args)
        return dict(directory=str(args.output_dir), best_epoch=1, validation_action_accuracy=0.5)

    monkeypatch.setattr(bc, "load_demonstrations", lambda path: dict())
    monkeypatch.setattr(bc, "train_one", fake_train)
    args = SimpleNamespace(seed=42, epochs=2, data_dir=tmp_path / "demos", output_dir=tmp_path / "bc")
    result = bc.train(args)
    assert result["agents"] == EXPECTED_AGENTS
    assert [a.seed for a in captured] == list(range(42, 42 + EXPECTED_AGENTS))
    assert len(set(a.data for a in captured)) == EXPECTED_AGENTS
    assert len(set(a.output_dir for a in captured)) == EXPECTED_AGENTS
    assert args.seed == 42


def test_default_paths_are_variant_specific():
    assert bc.parser().parse_args(["collect"]).output_dir.parts[-2] == bc.VARIANT
    assert bc.parser().parse_args(["train"]).data_dir.parts[-2] == bc.VARIANT


def test_bad_demonstration_shapes(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, observations=np.zeros((4, 3)), actions=np.array([0, 1, 0, 1]),
             episode_ids=np.array([0, 0, 1, 1]), episode_seeds=np.array([0, 1]))
    with pytest.raises(ValueError):
        bc.load_demonstrations(path)
