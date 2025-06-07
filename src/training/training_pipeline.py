import os
from pathlib import Path
import torch
from datasets import DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    PreTrainedTokenizerBase,
)
from typing import Optional

from src.config import ConfigManager
from src.utils import (
    TokenizerInitializer,
    ModelInitializer,
    MemoryManager,
    AppLogger,
    CheckpointUtils,
)
from src.type_defs import (
    ArgLoggerType,
    InitializedModelType,
    ModelCLIType,
)
from .dataset_manager import DatasetManager
from .metrics_calculator import MetricsCalculator
from .custom_callbacks import CustomEarlyStoppingCallback, LoggingCallback

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class TrainingPipeline:
    def __init__(
        self, config: Optional[ConfigManager] = None, logger: ArgLoggerType = None
    ):
        self.config = config or ConfigManager()
        self.logger = logger or AppLogger(name="training_pipeline").get_logger

        self.memory_manager = MemoryManager(self.logger)
        self.dataset_manager = DatasetManager(self.config, self.logger)
        self.tokenizer_initializer = TokenizerInitializer(self.config, self.logger)
        self.model_initializer = ModelInitializer(self.config, self.logger)
        self.checkpoint_utils = CheckpointUtils(self.logger)

    def _configure_training_args(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Seq2SeqTrainingArguments:
        base_args = self.config.training_config.to_dict()

        generation_config = GenerationConfig(
            max_length=self.config.dataset_config.max_training_length,
            decoder_start_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(  # type:ignore
                self.config.lang_config.tgt_nllb_lang_code
            ),
            **self.config.generation_config.to_dict(),
        )

        output_dir = self.config.model_config.checkpoints_path

        return Seq2SeqTrainingArguments(
            **base_args,  # type:ignore
            output_dir=str(output_dir),
            generation_config=generation_config,
        )

    def _initialize_trainer(
        self,
        model: InitializedModelType,
        tokenizer: PreTrainedTokenizerBase,
        tokenized_dataset: DatasetDict,
        training_args: Seq2SeqTrainingArguments,
        data_collator: DataCollatorForSeq2Seq,
    ) -> Seq2SeqTrainer:
        """
        Initializes a Hugging Face trainer object with configured training arguments, data collator, and metrics calculator.

        Args:
            model (InitializedModelType): The model object to be trained.
            tokenizer (PreTrainedTokenizerBase): The tokenizer object used for tokenization.
            tokenized_dataset (DatasetDict): The tokenized dataset to be used for training and evaluation.
            training_args (Seq2SeqTrainingArguments): The training arguments configuration.
            data_collator (DataCollatorForSeq2Seq): The data collator for batching and padding.

        Returns:
            A Seq2SeqTrainer object.
        """
        metrics_calculator = MetricsCalculator(self.config, tokenizer, self.logger)

        return Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],  # type:ignore
            data_collator=data_collator,
            compute_metrics=metrics_calculator.compute_metrics,
            callbacks=[
                LoggingCallback(self.logger),
                CustomEarlyStoppingCallback(
                    early_stopping_patience=5,
                    early_stopping_threshold=0.001,
                    logger=self.logger,
                ),
            ],
        )

    def _save_model(
        self, trainer: Seq2SeqTrainer, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """
        Saves the model and tokenizer to the result directory.

        Args:
            trainer (Seq2SeqTrainer): The trainer object.
            tokenizer (PreTrainedTokenizerBase): The tokenizer object.
        """
        output_dir: Path = self.config.model_config.result_path

        trainer.model.save_pretrained(output_dir)  # type:ignore
        tokenizer.save_pretrained(output_dir)
        trainer.model.config.save_pretrained(output_dir)  # type:ignore

        self.logger.info(f"📦 The model and tokenizer are stored in {output_dir}")

    def run_training(
        self,
        model_cli_path: ModelCLIType = None,
        with_lora: bool = False,
    ) -> None:
        self.logger.info("🚀 Starting training pipeline")

        model = None
        tokenizer = None

        try:
            self.memory_manager.clear()

            # 1. Initialize tokenizer
            tokenizer = self.tokenizer_initializer.initialize()

            # 2. Initialize model
            model = self.model_initializer.initialize(
                for_training=True,
                torch_dtype=torch.float16,
                model_cli_path=model_cli_path,
                with_lora=with_lora,
            )

            # 3. Load dataset
            dataset = self.dataset_manager.get_dataset()
            tokenized_dataset = self.dataset_manager.tokenize_dataset(
                dataset, tokenizer
            )

            # 4. Configure training arguments
            training_args = self._configure_training_args(tokenizer)

            data_collator = DataCollatorForSeq2Seq(tokenizer, model)

            self.memory_manager.clear()

            # 5. Initialize trainer
            trainer = self._initialize_trainer(
                model, tokenizer, tokenized_dataset, training_args, data_collator
            )

            # 6. Determine the checkpoint to resume from
            checkpoint_dir = None
            if model_cli_path and "checkpoint-" in model_cli_path:
                # 6.1. Extract the checkpoint name from model_path
                checkpoint_name = Path(model_cli_path).name
                checkpoint_dir = self.checkpoint_utils.get_checkpoint_path(
                    model_config=self.config.model_config,
                    checkpoint=checkpoint_name,
                )
            else:
                checkpoint_dir = self.config.model_config.last_checkpoint

            self.logger.info(
                f"Training will start from {'checkpoint ' + str(checkpoint_dir) if checkpoint_dir else 'the beginning'}"
            )

            # 7. Start training
            trainer.train(
                resume_from_checkpoint=str(checkpoint_dir) if checkpoint_dir else None
            )
            self._save_model(trainer, tokenizer)

            self.memory_manager.clear()

            self.logger.info("✅ Training pipeline completed successfully")
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            raise
        except Exception as e:
            self.logger.error(f"❌ Caught error at training pipeline: {str(e)}")
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
