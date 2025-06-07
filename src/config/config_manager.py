from typing import Optional
from .dataset import DatasetConfig
from .generation import GenerationConfigParams
from .language import LanguageConfig
from .logging import LoggingConfig
from .lora import LoraConfig
from .models import ModelConfig
from .paths import BasePathConfig, TranslationPathConfig, TrainingPathConfig
from .training import TrainingConfig
from .translation import TranslationConfig

from src.type_defs import LangCode


class ConfigManager:
    def __init__(
        self,
        src_lang: LangCode = LangCode.EN,
        tgt_lang: LangCode = LangCode.RU,
        input_file: Optional[str] = None,
    ):
        self.dataset_config = DatasetConfig()
        self.generation_config = GenerationConfigParams()
        self.base_path_config = BasePathConfig()
        self.translation_path_config = TranslationPathConfig(input_file=input_file)
        self.training_path_config = TrainingPathConfig()
        self.lang_config = LanguageConfig(src_lang, tgt_lang)
        self.lora_config = LoraConfig()
        self.logging_config = LoggingConfig()
        self.model_config = ModelConfig(self.training_path_config)
        self.training_config = TrainingConfig(str(self.base_path_config.logging_dir))
        self.translation_config = TranslationConfig(
            self.translation_path_config, self.lang_config
        )
