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

    num_beams: int = 6
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True

    def to_dict(self) -> GenerationConfigType:
        return asdict(self)

    def to_generation_config(self) -> GenerationConfig:
        return GenerationConfig(**self.to_dict())
