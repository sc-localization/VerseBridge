from dataclasses import dataclass

from src.type_defs import TokenizerConfigType


@dataclass
class DatasetConfig:
    max_model_length: int = 1024  # tokenizer.model_max_length
    max_training_length: int = 128
    data_split_ratio: float = 0.9

    def to_dict(self) -> TokenizerConfigType:
        return {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_model_length,
        }
