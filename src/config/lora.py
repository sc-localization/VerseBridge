from dataclasses import dataclass, field
from peft import TaskType, LoraConfig as PeftLoraConfig

from src.type_defs import LoraTargetModulesType


@dataclass
class LoraConfig(PeftLoraConfig):
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: LoraTargetModulesType = field(
        default_factory=lambda: ["q", "v"]
    )
    task_type = TaskType.SEQ_2_SEQ_LM
    bias = "none"
