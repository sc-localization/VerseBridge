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
            "max_length": int(
                self.max_model_length / 4
            ),  # when using the original value, the model starts generating a lot of nonsensical text and loses context
        }
