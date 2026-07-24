from __future__ import annotations

from pathlib import Path

from models.comprehensive_config import (
    DEVELOPMENT_END_SEASON,
    GAME_TARGETS,
    PLAYER_TARGETS,
    TEST_SEASONS,
    VALIDATION_SEASON,
)
from models.comprehensive_core import all_features, json_safe


def test_chronological_periods_do_not_overlap() -> None:
    assert DEVELOPMENT_END_SEASON < VALIDATION_SEASON
    assert all(season > VALIDATION_SEASON for season in TEST_SEASONS)
    assert TEST_SEASONS == sorted(set(TEST_SEASONS))


def test_current_targets_are_not_direct_model_features() -> None:
    for config in PLAYER_TARGETS.values():
        features = all_features(config["groups"])
        assert config["target"] not in features
    for config in GAME_TARGETS.values():
        features = all_features(config["groups"])
        assert config["target"] not in features


def test_feature_lists_are_unique() -> None:
    for config in [*PLAYER_TARGETS.values(), *GAME_TARGETS.values()]:
        features = all_features(config["groups"])
        assert len(features) == len(set(features))


def test_baselines_are_pregame_values() -> None:
    for config in PLAYER_TARGETS.values():
        assert config["baseline"].endswith("_shrunk_baseline")
        assert config["baseline"] != config["target"]
    assert GAME_TARGETS["game_margin"]["baseline"] == "spread_line"
    assert GAME_TARGETS["game_total"]["baseline"] == "total_line"


def test_fragmented_runner_reconstructs_and_compiles() -> None:
    root = Path(__file__).resolve().parents[1]
    fragments = sorted((root / "models" / "comprehensive_parts").glob("*.inc"))
    assert len(fragments) >= 4
    source = "".join(fragment.read_text(encoding="utf-8") for fragment in fragments)
    compile(source, "run_comprehensive_generated.py", "exec")
    assert "def walk_forward" in source
    assert "locked_first_pass" in source


def test_json_safe_removes_non_finite_values() -> None:
    assert json_safe(float("nan")) is None
    assert json_safe(float("inf")) is None
