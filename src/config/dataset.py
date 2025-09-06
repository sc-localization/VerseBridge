from dataclasses import dataclass

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
