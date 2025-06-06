from dataclasses import dataclass, field
from pathlib import Path


from src.type_defs import ModelNameType, LastCheckpointPathType
from .paths import TrainingPathConfig


@dataclass
class ModelConfig:
    path_config: TrainingPathConfig
    model_name: ModelNameType = "facebook/nllb-200-distilled-1.3B"
    result_path: Path = field(init=False)
    checkpoints_path: Path = field(init=False)
    last_checkpoint: LastCheckpointPathType = field(init=False, default=None)
