from dataclasses import dataclass, field
from peft import TaskType, LoraConfig as PeftLoraConfig

from src.type_defs import LoraTargetModulesType


@dataclass
class LoraConfig(PeftLoraConfig):
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: LoraTargetModulesType = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    task_type = TaskType.SEQ_2_SEQ_LM
