from typing import Dict, List
from datasets import load_dataset, DatasetDict

from transformers import PreTrainedTokenizerBase, BatchEncoding

from src.config import ConfigManager
from src.type_defs import LoggerType


class DatasetManager:
    def __init__(self, config: ConfigManager, logger: LoggerType):
        self.config = config
        self.logger = logger

    def get_dataset(
        self,
    ) -> DatasetDict:
        """Loads the dataset from JSON files specified in the config and returns it as a DatasetDict.

        Returns:
            DatasetDict: A dictionary containing the train and test datasets.
        """
        data_files = {
            "train": str(self.config.training_path_config.json_files["train"]),
            "test": str(self.config.training_path_config.json_files["test"]),
        }

        dataset: DatasetDict = load_dataset("json", data_files=data_files)  # type: ignore

        if not dataset["train"] or not dataset["test"]:
            self.logger.error("One of the datasets is empty!")
            raise ValueError("Dataset is empty")

        self.logger.debug(f"Example from validation dataset: {dataset['test'][0]}")

        return dataset

    def tokenize_dataset(
        self, dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase
    ) -> DatasetDict:
        """Tokenizes the dataset using the provided tokenizer.

        Args:
            dataset (DatasetDict): The dataset to tokenize.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.

        Returns:
            DatasetDict: The tokenized dataset.
        """
        tokenizer_base_args = self.config.dataset_config.to_dict()

        def tokenize_function(examples: Dict[str, List[str]]) -> BatchEncoding:
            """
            Tokenizes the input examples and prepares them for the model.
            """
            tokenizer.src_lang = self.config.lang_config.src_nllb_lang_code
            inputs = tokenizer(
                examples["original"],
                **tokenizer_base_args,
            )
            tokenizer.src_lang = self.config.lang_config.tgt_nllb_lang_code
            targets = tokenizer(
                examples["translated"],
                **tokenizer_base_args,
            )
            inputs["labels"] = targets["input_ids"]
            inputs["decoder_input_ids"] = targets["input_ids"]

            return inputs

        try:
            tokenized_dataset: DatasetDict = dataset.map(
                tokenize_function, batched=True, 
                remove_columns=["original", "translated"]
            )
            self.logger.debug("Dataset tokenized successfully")

            return tokenized_dataset
        except Exception as e:
            self.logger.error(f"Failed to tokenize dataset: {str(e)}")
            raise
