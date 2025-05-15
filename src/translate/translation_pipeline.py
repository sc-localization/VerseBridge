import os
import torch
from typing import Optional


from src.config import ConfigManager
from src.utils import (
    AppLogger,
    FileUtils,
    MemoryManager,
    ModelInitializer,
    TokenizerInitializer,
)
from src.type_defs import ArgLoggerType, ModelCLIType, TranslatedFileNameType
from .text_processor import TextProcessor
from .translator import Translator
from .ini_file_processor import IniFileProcessor


class TranslationPipeline:
    def __init__(
        self, config: Optional[ConfigManager] = None, logger: ArgLoggerType = None
    ):
        self.config = config or ConfigManager()
        self.logger = logger or AppLogger("translation_pipeline").get_logger

        self.file_utils = FileUtils(self.logger)
        self.memory_manager = MemoryManager(self.logger)
        self.tokenizer_initializer = TokenizerInitializer(self.config, self.logger)
        self.text_processor = TextProcessor(
            self.config.translation_config.protected_patterns
        )
        self.model_initializer = ModelInitializer(self.config, self.logger)
        self.ini_file_processor = IniFileProcessor(
            self.config, self.text_processor, self.logger
        )

    def run_translation(
        self,
        translated_file_name: TranslatedFileNameType = None,
        model_cli_path: ModelCLIType = None,
    ) -> None:
        self.logger.info("🚀 Starting translation pipeline")
        self.memory_manager.clear()

        try:
            if not os.path.exists(self.config.translation_config.source_ini_file_path):
                self.logger.error(
                    f"Source file {self.config.translation_config.source_ini_file_path} does not exist"
                )
                raise FileNotFoundError(
                    f"Source file {self.config.translation_config.source_ini_file_path} does not exist"
                )

            # 1. Copying source file
            self.file_utils.copy_files(
                self.config.translation_config.source_ini_file_path,
                self.config.translation_config.translate_src_dir,
            )

            # 2. Initialize model
            model = self.model_initializer.initialize(
                for_training=False,
                torch_dtype=torch.float16,
                model_cli_path=model_cli_path,
            )

            # 3. Initialize tokenizer
            tokenizer = self.tokenizer_initializer.initialize()

            # 4. Initializing translator
            translator = Translator(
                self.config, model, tokenizer, self.text_processor, self.logger
            ).create_translator()

            # 5. Creating destination directory
            os.makedirs(
                self.config.translation_config.translate_dest_dir, exist_ok=True
            )

            # 6. Getting INI files
            ini_files = [
                f
                for f in os.listdir(self.config.translation_config.translate_src_dir)
                if f.endswith(".ini")
            ]

            if not ini_files:
                self.logger.error("No INI files found in source directory")
                raise FileNotFoundError("No INI files found")

            # 7. Translating INI files
            for file_name in ini_files:
                source_path = (
                    self.config.translation_config.translate_src_dir / file_name
                )

                destination_path = self.config.translation_config.translate_dest_dir / (
                    translated_file_name or file_name
                )

                self.ini_file_processor.translate_file(
                    source_path, destination_path, translator
                )

            self.logger.info("✅ Translation pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Caught error at translation pipeline: {str(e)}")
            raise
