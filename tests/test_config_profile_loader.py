import json
from pathlib import Path

import pytest

from src.config.config_profile_loader import ConfigProfileLoader
from src.type_defs import (
    LabelNames,
    LogReportTarget,
    Metric,
    Optimizer,
    Scheduler,
    Strategy,
)


@pytest.fixture
def default_config_path() -> Path:
    return Path("configs/default.json")


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    config = {
        "training": {
            "num_train_epochs": 5,
            "learning_rate": 1e-4,
            "optim": "adafactor",
            "lr_scheduler_type": "cosine",
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "metric_for_best_model": "bleu",
            "report_to": ["tensorboard"],
            "label_names": ["labels"],
        },
        "generation": {
            "num_beams": 4,
        },
    }
    path = tmp_path / "test_config.json"
    path.write_text(json.dumps(config))
    return path


def test_load_default_config(default_config_path: Path):
    """Loading the default config populates all sections."""
    loader = ConfigProfileLoader(default_config_path).load()

    training = loader.get_section("training")
    assert training["num_train_epochs"] == 10
    assert training["optim"] == Optimizer.paged_adamw_8bit

    generation = loader.get_section("generation")
    assert generation["num_beams"] == 2

    translation = loader.get_section("translation")
    assert translation["batch_size"] == 32

    dataset = loader.get_section("dataset")
    assert dataset["max_model_length"] == 1024


def test_partial_override_preserves_other_sections(tmp_config: Path):
    """A config with only some sections leaves other sections empty."""
    loader = ConfigProfileLoader(tmp_config).load()

    training = loader.get_section("training")
    assert training["num_train_epochs"] == 5
    assert training["learning_rate"] == 1e-4

    # Sections not in the partial config return empty dicts
    translation = loader.get_section("translation")
    assert translation == {}


def test_enum_string_conversion(tmp_config: Path):
    """String enum values in JSON are converted to Python enums."""
    loader = ConfigProfileLoader(tmp_config).load()
    training = loader.get_section("training")

    assert training["optim"] == Optimizer.adafactor
    assert training["lr_scheduler_type"] == Scheduler.cosine
    assert training["eval_strategy"] == Strategy.EPOCH
    assert training["save_strategy"] == Strategy.EPOCH
    assert training["logging_strategy"] == Strategy.STEPS
    assert training["metric_for_best_model"] == Metric.BLEU
    assert training["report_to"] == [LogReportTarget.TENSORBOARD]
    assert training["label_names"] == [LabelNames.LABELS]


def test_generation_override(tmp_config: Path):
    """Generation section values are loaded correctly."""
    loader = ConfigProfileLoader(tmp_config).load()
    generation = loader.get_section("generation")

    assert generation["num_beams"] == 4


def test_missing_file_raises_error(tmp_path: Path):
    """Explicitly providing a non-existent path raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError, match="Config profile not found"):
        ConfigProfileLoader(missing).load()


def test_no_config_no_default_uses_dataclass_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When no config_path and no default.json exists, returns empty sections."""
    monkeypatch.chdir(tmp_path)

    loader = ConfigProfileLoader(None).load()

    assert loader.get_section("training") == {}
    assert loader.get_section("generation") == {}
    assert loader.get_section("translation") == {}


def test_invalid_json_structure_raises_error(tmp_path: Path):
    """JSON that is not Dict[str, Dict[str, Any]] raises ValueError."""
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match="Invalid config profile structure"):
        ConfigProfileLoader(bad_config).load()


def test_invalid_nested_structure_raises_error(tmp_path: Path):
    """JSON with non-dict section values raises ValueError."""
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps({"training": "not_a_dict"}))

    with pytest.raises(ValueError, match="Invalid config profile structure"):
        ConfigProfileLoader(bad_config).load()


def test_unknown_sections_logged_but_not_error(tmp_path: Path):
    """Unknown sections produce a warning but don't raise."""
    config = tmp_path / "extra.json"
    config.write_text(json.dumps({"unknown_section": {"key": "value"}}))

    loader = ConfigProfileLoader(config).load()
    assert loader.get_section("unknown_section") == {"key": "value"}


def test_ner_training_enum_conversion(tmp_path: Path):
    """NER training section enum fields are converted."""
    config = tmp_path / "ner.json"
    config.write_text(json.dumps({
        "ner_training": {
            "eval_strategy": "epoch",
        }
    }))

    loader = ConfigProfileLoader(config).load()
    ner_training = loader.get_section("ner_training")
    assert ner_training["eval_strategy"] == Strategy.EPOCH
