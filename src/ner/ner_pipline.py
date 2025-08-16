import subprocess
from typing import Optional

from src.config import ConfigManager
from src.utils import (
    AppLogger,
    FileUtils,
    MemoryManager,
    TokenizerInitializer,
    ModelInitializer,
)
from src.preprocess import DataSplitter
from src.type_defs import ArgLoggerType, is_json_data_ner_list_type

from .extract_ner import NERExtractor
from .convert_to_bio import BIOConverter
from .ner_training import NERTraining


class NERPipeline:
    def __init__(
        self, config: Optional[ConfigManager] = None, logger: ArgLoggerType = None
    ):
        self.config = config or ConfigManager()
        self.logger = logger or AppLogger("ner_pipeline").get_logger

        self.file_utils = FileUtils(self.logger)
        self.memory_manager = MemoryManager(self.logger)

        self.model_initializer = ModelInitializer(self.config, self.logger, task="ner")
        self.tokenizer_initializer = TokenizerInitializer(
            self.config, self.logger, task="ner"
        )

        self.data_splitter = DataSplitter(
            self.config.dataset_config.data_split_ratio, self.logger
        )

        self.tokenizer = self.tokenizer_initializer.initialize()

        # Инициализация модели для инференса (NERExtractor)
        self.base_model = self.model_initializer.initialize(for_training=False)

        # Инициализация модели для обучения (NERTraining)
        self.training_model = self.model_initializer.initialize(for_training=True)

    def run_extraction(self):
        """Runs NER data extraction."""
        self.logger.info("🚀 Starting NER extraction")
        self.memory_manager.clear()

        try:
            self.ner_extractor = NERExtractor(
                self.config,
                self.base_model,
                self.tokenizer,
                self.logger,
            )
            self.ner_extractor.run()

            self.logger.info("✅ NER extraction completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Error during NER extraction: {str(e)}")
            raise

    def run_review(self):
        """Runs the Streamlit application for reviewing NER data."""
        self.logger.info("🚀 Starting NER review")

        unannotated_path = self.config.ner_path_config.extracted_ner_data_path
        if not unannotated_path.exists():
            self.logger.warning(
                f"Unannotated NER file not found: {unannotated_path}. Running extraction step automatically."
            )
            self.run_extraction()

        try:
            subprocess.run(["streamlit", "run", "src/ner/review_ner_streamlit_app.py"])
            self.logger.info("✅ NER review completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Error during NER review: {str(e)}")
            raise

    def run_bio_conversion(self):
        """Converts NER data to BIO format and splits into train/test."""
        self.logger.info("🚀 Starting BIO conversion")
        self.memory_manager.clear()

        corrected_path = self.config.ner_path_config.corrected_streamlit_data_path
        unannotated_path = self.config.ner_path_config.extracted_ner_data_path

        try:
            # 1. Load NER data
            if not corrected_path.exists():
                self.logger.warning(
                    f"⚠️ Corrected dataset file not found: {corrected_path}. "
                    f"Using unannotated dataset from {unannotated_path}. "
                    f"Or you can stop and running review step."
                )
                ner_data = self.file_utils.load_json(unannotated_path)
            else:
                self.logger.info(f"Loading corrected NER data from {corrected_path}")
                ner_data = self.file_utils.load_json(corrected_path)

            if not is_json_data_ner_list_type(ner_data):
                raise ValueError("Invalid NER data format")

            # 2. Convert to BIO format and save
            self.bio_converter = BIOConverter(self.config, self.tokenizer, self.logger)
            bio_data = self.bio_converter.convert_to_bio(ner_data)

            self.file_utils.save_json(
                bio_data, self.config.ner_path_config.bio_ner_dataset_path
            )

            # 3. Split into train and test data
            train_data, test_data = self.data_splitter.split_data(bio_data)

            self.file_utils.save_json(
                train_data, self.config.ner_path_config.trained_data_files["train"]
            )
            self.file_utils.save_json(
                test_data, self.config.ner_path_config.trained_data_files["test"]
            )

            self.logger.info(
                "✅ BIO conversion and data splitting completed successfully"
            )
        except Exception as e:
            self.logger.error(f"❌ Error during BIO conversion: {str(e)}")
            raise

    def run_training(self):
        """Runs NER model training in English."""
        self.logger.info("🚀 Starting NER training")
        self.memory_manager.clear()

        train_path = self.config.ner_path_config.trained_data_files["train"]
        test_path = self.config.ner_path_config.trained_data_files["test"]

        if not train_path.exists() or not test_path.exists():
            self.logger.warning(
                f"Training files not found. Running bio conversion step automatically."
            )
            self.run_bio_conversion()

        try:
            self.ner_trainer = NERTraining(
                self.config, self.training_model, self.tokenizer, self.logger
            )
            self.ner_trainer.train(train_path, test_path)

            self.logger.info("✅ NER training completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Error during NER training: {str(e)}")
            raise
