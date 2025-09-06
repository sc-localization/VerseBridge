from dataclasses import dataclass, replace

from src.type_defs import TokenizerOptionsType, TokenizerConfigType


@dataclass
class DatasetConfig:
    max_model_length: int = 1024  # tokenizer.model_max_length
    max_training_length: int = 128
    data_split_ratio: float = 0.9

    @classmethod
    def _to_dict(cls) -> TokenizerOptionsType:
        return {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }

    @property
    def translation_dict(self) -> TokenizerConfigType:
        return {
            **self._to_dict(),
            "max_length": int(self.max_model_length),
        }

    @property
    def training_dict(self) -> TokenizerConfigType:
        return {
            **self._to_dict(),
            "max_length": int(self.max_training_length),
        }

    def update_max_training_length(self, new_value: int) -> "DatasetConfig":
        if new_value <= 0:
            raise ValueError("max_training_length must be positive")

        if new_value > self.max_model_length:
            raise ValueError(
                f"max_training_length ({new_value}) cannot exceed max_model_length ({self.max_model_length})"
            )

        return replace(self, max_training_length=new_value)
