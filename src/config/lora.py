from dataclasses import dataclass, field
from typing import Optional
from peft import TaskType, LoraConfig as PeftLoraConfig

from src.type_defs import LoraTargetModulesType


@dataclass
class LoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: LoraTargetModulesType

    def to_peft_config(self, task_type: Optional[TaskType] = TaskType.SEQ_2_SEQ_LM) -> PeftLoraConfig:
        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            task_type=task_type,
            bias="none",
        )
