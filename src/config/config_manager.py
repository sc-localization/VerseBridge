from typing import Optional

from .dataset import DatasetConfig
from .generation import GenerationConfigParams
from .language import LanguageConfig
from .logging import LoggingConfig
from .lora import LoraConfig
from .models import ModelConfig
from .paths import (
    BasePathConfig,
    NERPathConfig,
    TranslationPathConfig,
    TrainingPathConfig,
)
from .ner import NERConfig
from .training import TranslationTrainingConfig, NerTrainingConfig
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
        self.translation_path_config = TranslationPathConfig(
            src_lang, tgt_lang, input_file=input_file
        )
        self.training_path_config = TrainingPathConfig(src_lang, tgt_lang)
        self.ner_path_config = NERPathConfig(input_file=input_file)

        self.lang_config = LanguageConfig(src_lang, tgt_lang)

        self.lora_config = LoraConfig()

        self.logging_config = LoggingConfig()
        self.model_config = ModelConfig(self.training_path_config)

        self.training_config = TranslationTrainingConfig(
            str(self.base_path_config.logging_dir)
        )
        self.translation_config = TranslationConfig()

        self.ner_config = NERConfig(
            training_config=NerTrainingConfig(str(self.base_path_config.logging_dir))
        )
