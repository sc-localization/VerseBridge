import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from src.type_defs import (
    ConfigJSONType,
    is_config_json_type,
    LabelNames,
    LogReportTarget,
    Metric,
    Optimizer,
    Scheduler,
    Strategy,
)

_ENUM_FIELDS: Dict[str, Dict[str, Type[Any]]] = {
    "training": {
        "optim": Optimizer,
        "lr_scheduler_type": Scheduler,
        "eval_strategy": Strategy,
        "save_strategy": Strategy,
        "logging_strategy": Strategy,
        "metric_for_best_model": Metric,
    },
    "ner_training": {
        "eval_strategy": Strategy,
    },
}

_LIST_ENUM_FIELDS: Dict[str, Dict[str, Type[Any]]] = {
    "training": {
        "report_to": LogReportTarget,
        "label_names": LabelNames,
    },
}

_VALID_SECTIONS = frozenset(
    {"training", "ner_training", "translation", "generation", "lora", "dataset", "ner"}
)


class ConfigProfileLoader:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path
        self._default_path = Path("configs/default.json")
        self._logger = logging.getLogger("config_profile_loader")
        self._data: ConfigJSONType = {}

    def load(self) -> "ConfigProfileLoader":
        path = self._resolve_path()

        if path is None:
            self._logger.info("No config profile found, using dataclass defaults")
            return self

        self._logger.info(f"Loading config profile from {path}")

        raw = self._read_json(path)

        if not is_config_json_type(raw):
            raise ValueError(
                f"Invalid config profile structure in {path}: "
                "expected a JSON object with section names mapping to objects"
            )

        self._validate(raw, path)
        self._data = raw
        self._convert_enums()

        return self

    def get_section(self, name: str) -> Dict[str, Any]:
        return dict(self._data.get(name, {}))

    def _resolve_path(self) -> Optional[Path]:
        if self._config_path is not None:
            if not self._config_path.exists():
                raise FileNotFoundError(
                    f"Config profile not found: {self._config_path}"
                )
            return self._config_path

        if self._default_path.exists():
            return self._default_path

        return None

    # TODO: use file utils
    def _read_json(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _validate(self, data: Any, path: Path) -> None:
        unknown = set(data.keys()) - _VALID_SECTIONS
        if unknown:
            self._logger.warning(
                f"Unknown config sections in {path} will be ignored: {', '.join(sorted(unknown))}"
            )

    def _convert_enums(self) -> None:
        for section_name, field_map in _ENUM_FIELDS.items():
            section = self._data.get(section_name)
            if section is None:
                continue

            for field_name, enum_cls in field_map.items():
                if field_name in section:
                    section[field_name] = enum_cls(section[field_name])

        for section_name, field_map in _LIST_ENUM_FIELDS.items():
            section = self._data.get(section_name)
            if section is None:
                continue

            for field_name, enum_cls in field_map.items():
                if field_name in section:
                    section[field_name] = [enum_cls(v) for v in section[field_name]]
