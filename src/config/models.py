from dataclasses import dataclass, field
from pathlib import Path


from src.type_defs import TranslationModelNameType, LastCheckpointPathType
from .paths import TrainingPathConfig


@dataclass
class ModelConfig:
    path_config: TrainingPathConfig
    model_name: TranslationModelNameType = "google/madlad400-3b-mt"
    result_path: Path = field(init=False)
    checkpoints_path: Path = field(init=False)
    last_checkpoint: LastCheckpointPathType = field(init=False, default=None)
