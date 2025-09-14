from typing import Dict, List
from datasets import load_dataset, DatasetDict

from transformers import PreTrainedTokenizerBase, BatchEncoding

from src.config import ConfigManager
from src.type_defs import LoggerType, JsonTrainedDataFilePathsType


class DatasetManager:
    def __init__(self, config: ConfigManager, logger: LoggerType):
        self.config = config
        self.logger = logger

    def get_recommended_max_length(
        self,
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        sample_size: int = 1000,
    ) -> int:
        """Determines the recommended max_length for the tokenizer based on the dataset.

        Args:
            dataset (DatasetDict): The dataset to analyze.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
            sample_size (int, optional): The number of samples to use for analysis. Defaults to 1000.

        Returns:
            int: The recommended max_length.
        """
        self.logger.debug("🔍 Analyzing sequence lengths to recommend max_length...")

        ds_train = dataset["train"]
        ds_test = dataset["test"]
        sample_size = min(sample_size, min(len(ds_train), len(ds_test)))
        sample_train = ds_train.shuffle(seed=42).select(range(sample_size))
        sample_test = ds_test.shuffle(seed=42).select(range(sample_size))

        sample_list: List[Dict[str, str]] = (
            sample_train.to_list() + sample_test.to_list()
        )

        all_lengths: List[int] = [
            max(
                len(
                    tokenizer.encode(
                        f"{self.config.lang_config.tgt_lang_token} {item['original']}"
                    )
                ),
                len(tokenizer.encode(item["translated"])),
            )
            for item in sample_list
        ]

        sorted_lengths = sorted(all_lengths)
        total = len(sorted_lengths)

        p98_index = int(total * 0.98)
        p98_len = sorted_lengths[p98_index]

        recommended = ((p98_len + 7) // 8) * 8
        recommended = min(recommended, self.config.dataset_config.max_training_length) 

        self.logger.debug(f"📊 Sequence length statistics:")
        self.logger.debug(f"   98% texts shorter than: {p98_len}")
        self.logger.debug(f"🎯 Recommended max_length: {recommended}")

        return recommended

    def get_dataset(self, data_files: JsonTrainedDataFilePathsType) -> DatasetDict:
        """Loads the dataset from JSON files specified in the config and returns it as a DatasetDict.
        Args:
            data_files (JsonFilePathsType): Dictionary with paths to train and test JSON files.

        Returns:
            DatasetDict: A dictionary containing the train and test datasets.
        """

        dataset: DatasetDict = load_dataset("json", data_files={k: str(v) for k, v in data_files.items()})  # type: ignore

        if not dataset["train"] or not dataset["test"]:
            self.logger.error("One of the datasets is empty!")
            raise ValueError("Dataset is empty")

        self.logger.debug(
            f"Example from validation dataset: {dataset['test'][0]}\nTrain dataset size: {len(dataset['train'])}"
        )

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
        tokenizer_base_args = self.config.dataset_config.training_dict

        def tokenize_function(examples: Dict[str, List[str]]) -> BatchEncoding:
            """
            Tokenizes the input examples and prepares them for the model.
            """
            inputs = [
                f"{self.config.lang_config.tgt_lang_token} {example}"
                for example in examples["original"]
            ]
            targets = examples["translated"]

            model_inputs = tokenizer(
                inputs,
                **tokenizer_base_args,
            )
            labels = tokenizer(
                targets,
                **tokenizer_base_args,
            )
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        try:
            tokenized_dataset: DatasetDict = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["original", "translated"],
            )
            self.logger.debug("Dataset tokenized successfully")

            return tokenized_dataset
        except Exception as e:
            self.logger.error(f"Failed to tokenize dataset: {str(e)}")
            raise
