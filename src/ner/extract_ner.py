from tqdm import tqdm
from transformers import pipeline

from src.config import ConfigManager
from src.utils import FileUtils, TokenizerInitializer
from src.preprocess import TextCleaner
from src.type_defs import (
    LoggerType,
    INIDataType,
    JSONDataNERListType,
    EntitiesType,
    InitializedModelType,
    InitializedTokenizerType,
)


class NERExtractor:
    def __init__(
        self,
        config: ConfigManager,
        model: InitializedModelType,
        tokenizer: InitializedTokenizerType,
        logger: LoggerType,
    ):
        self.config = config
        self.logger = logger

        self.tokenizer = tokenizer
        self.model = model

        self.file_utils = FileUtils(self.logger)

        self.text_cleaner = TextCleaner(
            tokenizer=TokenizerInitializer(
                self.config, self.logger, task="translation"
            ).initialize(),
            protected_patterns=self.config.translation_config.protected_patterns,
            max_model_length=self.config.dataset_config.max_model_length,
            logger=self.logger,
        )

        self.ner = None

    def initialize_ner(self):
        try:
            self.ner = pipeline(
                task="token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy=self.config.ner_config.aggregation_strategy,
                batch_size=8,
            )
        except Exception as e:
            self.logger.error(f"Error initializing NER pipeline: {str(e)}")
            raise

    def extract_ner(self, ini_data: INIDataType) -> JSONDataNERListType:
        """Extracts NER entities from INI data."""

        dataset: JSONDataNERListType = []

        for key, raw_text in tqdm(ini_data.items(), desc="Extracting NER"):
            if key in self.config.translation_config.exclude_keys:
                continue

            cleaned_text = self.text_cleaner.clean_text(raw_text, remove_patterns=True)

            if cleaned_text and self.ner:
                threshold_confidence = self.config.ner_config.threshold_confidence
                entities: EntitiesType = [
                    {
                        "start": int(ent["start"]),
                        "end": int(ent["end"]),
                        "label": ent["entity_group"],
                    }
                    for ent in self.ner(cleaned_text)
                    if float(ent["score"]) > threshold_confidence
                ]

                dataset.append({"id": key, "text": cleaned_text, "entities": entities})

        return dataset

    def run(self):
        """Main method for extracting NER."""

        self.initialize_ner()

        if self.ner is None:
            self.logger.error("NER pipeline is not initialized")
            raise Exception("NER pipeline is not initialized")

        input_file_path = self.config.ner_path_config.input_file_path

        if not input_file_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        self.logger.info(f"Extracting NER from {input_file_path}")

        ini_data = self.file_utils.parse_ini_file(
            input_file_path, self.config.translation_config.exclude_keys
        )

        if not ini_data:
            self.logger.error(f"No valid INI data found in {input_file_path}")
            raise FileNotFoundError(f"No valid INI data found in {input_file_path}")

        dataset = self.extract_ner(ini_data)

        self.file_utils.save_json(
            dataset, self.config.ner_path_config.extracted_ner_data_path
        )

        self.logger.info(
            f"Saved NER dataset to {self.config.ner_path_config.extracted_ner_data_path}"
        )
