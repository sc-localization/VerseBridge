from dataclasses import asdict, dataclass
from transformers import GenerationConfig

from src.type_defs import GenerationConfigType


@dataclass(frozen=True)
class GenerationConfigParams:
    """
    Parameters for generation with Hugging Face Transformers models.
    More:
    - https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/text_generation#transformers.GenerationConfig
    - https://huggingface.co/docs/transformers/generation_strategies
    """

    num_beams: int = 4
    early_stopping: bool = True
    no_repeat_ngram_size=3
    repetition_penalty: float = 1.9

    def to_dict(self) -> GenerationConfigType:
        return asdict(self)

    def to_generation_config(self) -> GenerationConfig:
        return GenerationConfig(**self.to_dict())
