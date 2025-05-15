from dataclasses import asdict, dataclass

from src.type_defs import GenerationConfigType


@dataclass(frozen=True)
class GenerationConfig:
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
        return asdict(self)  # type: ignore
