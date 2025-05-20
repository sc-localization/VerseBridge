from typing import Optional

from src.config import ConfigManager
from src.utils import AppLogger, FileUtils, TokenizerInitializer
from src.type_defs import ArgLoggerType, is_json_data_list_type
from .ini_to_json_converter import IniConverter
from .json_cleaner import JsonCleaner
from .data_splitter import DataSplitter


class PreprocessPipeline:
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        logger: ArgLoggerType = None,
    ):
        self.logger = logger or AppLogger("preprocess_pipeline").get_logger
        self.config = config or ConfigManager()

        self.tokenizer_initializer = TokenizerInitializer(self.config, self.logger)
        self.tokenizer = self.tokenizer_initializer.initialize()

        self.ini_converter = IniConverter(
            self.config.translation_config.exclude_keys, self.logger
        )
        self.json_cleaner = JsonCleaner(
            self.tokenizer,
            self.config.translation_config.protected_patterns,
            self.config.dataset_config.max_model_length,
            self.logger,
        )
        self.data_splitter = DataSplitter(
            self.config.dataset_config.data_split_ratio, self.logger
        )

        self.file_utils = FileUtils(self.logger)

    def run_preprocess(self) -> None:
        self.logger.info("🚀 Starting preprocessing pipeline")

        try:
            original_ini = self.config.translation_config.original_ini_file_path
            translated_ini = self.config.path_config.ini_files["translated"]

            self.config.path_config.check_ini_files_exist()

            # Path to intermediate JSON files
            json_path = self.config.path_config.json_files
            data_json_path = json_path["data"]
            cleaned_json_path = json_path["cleaned"]
            train_json_path = json_path["train"]
            test_json_path = json_path["test"]

            # Step 1: Convert INI to JSON
            self.logger.info("🔃 Step 1: Converting INI files to JSON")

            json_data = self.ini_converter.create_json_data(original_ini, translated_ini)
            self.file_utils.save_json(json_data, data_json_path)

            # Step 2: Clean JSON-data
            self.logger.info("🧹 Step 2: Cleaning JSON data")

            loaded_json_data = self.file_utils.load_json(data_json_path)

            if not is_json_data_list_type(loaded_json_data):
                raise ValueError(
                    "JSON data must contain a list of dictionaries with 'original' and 'translated' keys"
                )

            cleaned_data = self.json_cleaner.clean_json_data(loaded_json_data)
            self.file_utils.save_json(cleaned_data, cleaned_json_path)

            # Step 3: Split data
            self.logger.info("🗃️ Step 3: Splitting data into train and test sets")

            loaded_cleaned_data = self.file_utils.load_json(cleaned_json_path)

            if not is_json_data_list_type(loaded_cleaned_data):
                raise ValueError(
                    "JSON data must contain a list of dictionaries with 'original' and 'translated' keys"
                )

            train_data, test_data = self.data_splitter.split_data(loaded_cleaned_data)
            self.file_utils.save_json(train_data, train_json_path)
            self.file_utils.save_json(test_data, test_json_path)

            self.logger.info("✅ Preprocessing pipeline completed successfully")
        except FileNotFoundError as e:
            self.logger.error(f"Missing required file: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Caught error at data preprocessing: {str(e)}")
            raise
