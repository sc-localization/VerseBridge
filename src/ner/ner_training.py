import os
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    BatchEncoding,
)
from pathlib import Path
from typing import Optional, Dict, Any, List
from evaluate import load as load_metric

from src.utils import (
    MemoryManager,
    CheckpointUtils,
)
from src.config import ConfigManager
from src.type_defs import LoggerType, InitializedModelType, InitializedTokenizerType
from src.training import DatasetManager

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class NERTraining:
    def __init__(
        self,
        config: ConfigManager,
        model: InitializedModelType,
        tokenizer: InitializedTokenizerType,
        logger: LoggerType,
    ) -> None:
        self.config = config
        self.logger = logger

        self.memory_manager = MemoryManager(self.logger)
        self.dataset_manager = DatasetManager(self.config, self.logger)
        self.checkpoint_utils = CheckpointUtils(self.logger)

        self.tokenizer = tokenizer
        self.model = model

    def load_dataset(self, train_path: Path, test_path: Path) -> DatasetDict:
        """Loads NER dataset using DatasetManager and formats it for NER."""

        self.logger.info(f"Loading NER dataset from {train_path} and {test_path}")

        try:
            dataset = self.dataset_manager.get_dataset(
                data_files={
                    "train": train_path,
                    "test": test_path,
                },
            )

            def format_dataset(
                raw_data: Dict[str, Any],
            ) -> Dict[str, Any]:
                tokens: List[str] = raw_data["tokens"]
                ner_tags: List[str] = raw_data["labels"]

                label_ids: List[int] = [
                    self.config.ner_config.label_list.index(tag) for tag in ner_tags
                ]

                return {"tokens": tokens, "labels": label_ids}

            dataset = dataset.map(format_dataset)

            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def tokenize_and_align_labels(self, raw_datasets: DatasetDict) -> DatasetDict:
        def tokenize_function(examples: Dict[str, Any]) -> BatchEncoding:
            tokenized_inputs = self.tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )
            all_labels = examples["labels"]
            new_labels = []

            for i, label in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids: List[int] = []
                previous_word_idx: Optional[int] = None

                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)

                    previous_word_idx = word_idx

                new_labels.append(label_ids)

            tokenized_inputs["labels"] = new_labels

            return tokenized_inputs

        return raw_datasets.map(tokenize_function, batched=True)

    def compute_metrics(self, p: Any) -> Dict[str, Any]:
        metric = load_metric("seqeval")

        predictions = p.predictions
        labels = p.label_ids
        predictions = np.argmax(predictions, axis=2)

        true_predictions: List[List[str]] = [
            [
                self.config.ner_config.label_list[pred_id]
                for pred_id, label_id in zip(pred, lab)
                if label_id != -100
            ]
            for pred, lab in zip(predictions, labels)
        ]

        true_labels: List[List[str]] = [
            [
                self.config.ner_config.label_list[label_id]
                for _, label_id in zip(pred, lab)
                if label_id != -100
            ]
            for pred, lab in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        if results is None:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }

        return {
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
            "accuracy": results.get("overall_accuracy", 0.0),
        }

    def train(self, train_path: Path, test_path: Path) -> None:
        """Trains the NER model."""

        self.logger.info(f"Training NER model on {train_path} and {test_path}")

        try:
            self.memory_manager.clear()

            raw_datasets = self.load_dataset(train_path, test_path)
            tokenized_datasets = self.tokenize_and_align_labels(raw_datasets)

            ner_cfg = self.config.ner_config.training_config

            if not ner_cfg:
                raise ValueError("NER training config is not specified")

            args = TrainingArguments(
                output_dir=str(self.config.ner_path_config.output_dir),
                logging_dir=str(self.config.ner_path_config.logging_dir),
                learning_rate=ner_cfg.learning_rate,
                per_device_train_batch_size=ner_cfg.per_device_train_batch_size,
                per_device_eval_batch_size=ner_cfg.per_device_eval_batch_size,
                num_train_epochs=ner_cfg.num_train_epochs,
                weight_decay=ner_cfg.weight_decay,
                save_total_limit=ner_cfg.save_total_limit,
                load_best_model_at_end=ner_cfg.load_best_model_at_end,
                metric_for_best_model=ner_cfg.metric_for_best_model,
                eval_strategy=ner_cfg.eval_strategy,
            )

            data_collator = DataCollatorForTokenClassification(self.tokenizer)

            trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )

            checkpoint_dir = self.checkpoint_utils.get_latest_checkpoint(
                self.config.ner_path_config.output_dir / "checkpoints"
            )

            trainer.train(
                resume_from_checkpoint=str(checkpoint_dir) if checkpoint_dir else None
            )

            trainer.save_model(str(self.config.ner_path_config.output_dir))
            self.tokenizer.save_pretrained(str(self.config.ner_path_config.output_dir))

            self.memory_manager.clear()

            self.logger.info(
                f"NER model saved to {self.config.ner_path_config.output_dir}"
            )
        except KeyboardInterrupt:
            self.logger.warning("NER Training interrupted by user")
            raise
        except Exception as e:
            self.logger.error(f"Error during NER training: {str(e)}")
            raise
        finally:
            self.memory_manager.clear()
