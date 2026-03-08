from pathlib import Path
from typing import Optional

from .config_profile_loader import ConfigProfileLoader
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
        config_path: Optional[Path] = None,
    ):
        profile = ConfigProfileLoader(config_path).load()

        self.dataset_config = DatasetConfig(**profile.get_section("dataset"))
        self.generation_config = GenerationConfigParams(**profile.get_section("generation"))

        self.base_path_config = BasePathConfig()
        self.translation_path_config = TranslationPathConfig(
            src_lang, tgt_lang, input_file=input_file
        )
        self.training_path_config = TrainingPathConfig(src_lang, tgt_lang)
        self.ner_path_config = NERPathConfig(input_file=input_file)

        self.lang_config = LanguageConfig(src_lang, tgt_lang)

        self.lora_config = LoraConfig(**profile.get_section("lora"))

        self.logging_config = LoggingConfig()
        self.model_config = ModelConfig(self.training_path_config)

        self.training_config = TranslationTrainingConfig(
            str(self.base_path_config.logging_dir),
            **profile.get_section("training"),
        )
        self.translation_config = TranslationConfig(**profile.get_section("translation"))

        ner_training_section = profile.get_section("ner_training")
        ner_section = profile.get_section("ner")
        self.ner_config = NERConfig(
            training_config=NerTrainingConfig(
                str(self.base_path_config.logging_dir),
                **ner_training_section,
            ),
            **ner_section,
        )
