import os
from pathlib import Path
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

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
        existing_translated_file: Optional[str] = None,
    ) -> None:
        self.logger.info("🚀 Starting translation pipeline")
        self.memory_manager.clear()

        model = None
        tokenizer = None

        try:
            self.config.path_config.check_input_ini_file_exists()

            input_ini_file_path = self.config.translation_config.input_ini_file_path
            translation_src_dir = self.config.translation_config.translation_src_dir
            translation_dest_dir = self.config.translation_config.translation_dest_dir

            existing_translated_ini_file_path = (
                Path(existing_translated_file) if existing_translated_file else None
            )

            # 1. Copying input `.ini` file
            self.file_utils.copy_files(input_ini_file_path, translation_src_dir)

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
            os.makedirs(translation_dest_dir, exist_ok=True)

            # 6. Getting INI files
            ini_files = [
                f for f in os.listdir(translation_src_dir) if f.endswith(".ini")
            ]

            if not ini_files:
                self.logger.error("No INI files found in source directory")
                raise FileNotFoundError("No INI files found")

            # 7. Translating INI files
            for file_name in ini_files:
                translation_input_file = translation_src_dir / file_name

                translation_result_file = translation_dest_dir / (
                    translated_file_name or file_name
                )

                self.ini_file_processor.translate_file(
                    translation_input_file,
                    translation_result_file,
                    translator,
                    existing_translated_file=existing_translated_ini_file_path,
                )

            self.logger.info("✅ Translation pipeline completed successfully")
        except KeyboardInterrupt:
            self.logger.warning("Translation interrupted by user")
            raise
        except Exception as e:
            self.logger.error(f"❌ Caught error at translation pipeline: {str(e)}")
            raise
        finally:
            self.logger.debug("Releasing model and tokenizer resources")

            if model is not None:
                model = None
                del model

            if tokenizer is not None:
                tokenizer = None
                del tokenizer

            self.memory_manager.clear()
