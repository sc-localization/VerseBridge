from .dataset import DatasetConfig
from .generation import GenerationConfigParams
from .language import LanguageConfig
from .logging import LoggingConfig
from .lora import LoraConfig
from .models import ModelConfig
from .paths import PathConfig
from .training import TrainingConfig
from .translation import TranslationConfig

from src.type_defs import LangCode, TranslationPriorityType


class ConfigManager:
    def __init__(
        self,
        src_lang: LangCode = LangCode.EN,
        tgt_lang: LangCode = LangCode.RU,
        input_file_path: str | None = None,
        translation_priority: TranslationPriorityType = "output",
    ):
        self.dataset_config = DatasetConfig()
        self.generation_config = GenerationConfigParams()
        self.path_config = PathConfig(input_file_path)
        self.lang_config = LanguageConfig(src_lang, tgt_lang)
        self.lora_config = LoraConfig()
        self.logging_config = LoggingConfig()
        self.model_config = ModelConfig(self.path_config)
        self.training_config = TrainingConfig(str(self.path_config.logging_dir))
        self.translation_config = TranslationConfig(
            self.path_config, self.lang_config, translation_priority
        )
