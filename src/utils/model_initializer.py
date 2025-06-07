import logging
import torch
from peft import PeftModel, get_peft_model, PeftMixedModel
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel
from pathlib import Path
from typing import Optional, Tuple

from src.config import ConfigManager
from src.type_defs import ModelCLIType, ModelPathOrName, InitializedModelType
from .checkpoint_utils import CheckpointUtils
from .logging_utils import LoggingUtils


class ModelInitializer:
    def __init__(self, config: ConfigManager, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logging_utils = LoggingUtils()
        self.model_config = config.model_config
        self.checkpoint_utils = CheckpointUtils(logger)

    def _load_base_model(
        self, model_name: ModelPathOrName, torch_dtype: torch.dtype = torch.float16
    ) -> PreTrainedModel:
        """
        Loads a base model with safety checks and error handling.

        Args:
            model_name (ModelPathOrName): The name of the model to load.
            torch_dtype (torch.dtype, optional): The data type for the model's tensors.
                Defaults to torch.float16.

        Returns:
            PreTrainedModel: The loaded sequence-to-sequence model.

        Raises:
            ValueError: If the model name is empty or if loading the model fails.
        """
        sanitized_name: str = self.logging_utils.sanitize_path_for_log(model_name)
        self.logger.info(f"Loading base model: {sanitized_name}")

        if not model_name:
            raise ValueError("Model name cannot be empty")

        try:
            return AutoModelForSeq2SeqLM.from_pretrained(  # type: ignore
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
            ).to(self.device)

        except Exception as e:
            error_msg: str = f"Failed to load base model {sanitized_name}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _apply_lora(self, model: PreTrainedModel) -> PeftModel | PeftMixedModel:
        """Applies LoRA configuration to the base model.

        Args:
            model (AutoModelForSeq2SeqLM): The base model to apply LoRA to.

        Returns:
            PeftModel | PeftMixedModel: The model with LoRA adapters.
        """
        if not hasattr(model, "enable_input_require_grads"):
            raise NotImplementedError("Model does not support LoRA adapters")

        lora_config = self.config.lora_config

        self.logger.info("Applying LoRA configuration")
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        return peft_model

    def _resolve_model_paths(
        self, use_lora: bool, model_path_or_name: Optional[ModelPathOrName] = None
    ) -> Tuple[Path, Path]:
        """
        Determines output paths for model artifacts based on configuration.

        Args:
            use_lora (bool): Whether to use LoRA adapters.
            model_path_or_name (Optional[ModelPathOrName]): The path to the model to load. If None, uses the
                base model.

        Returns:
            Tuple[Path, Path]: A tuple containing the output path for the model results and
                the output path for the model checkpoints.
        """
        if model_path_or_name and "checkpoint-" in model_path_or_name:
            # For checkpoints, use the parent directory
            model_dir = Path(model_path_or_name).parent.parent
            model_type = "checkpoint"
        else:
            model_type = "lora" if use_lora else "base_model"
            model_dir = self.config.base_path_config.models_dir / model_type

        result_path = model_dir / "result"
        checkpoints_path = model_dir / "checkpoints"

        self.logger.debug(
            f"Model paths for {model_type}: result={result_path}, checkpoints={checkpoints_path}"
        )

        return result_path, checkpoints_path

    def _handle_output_path_rollback(
        self, model_path_or_name: ModelPathOrName
    ) -> Tuple[str, bool, Path, Path]:
        """Handles path checking and fallback logic.

        Args:
            model_path_or_name (ModelPathOrName): The path to the model directory.

        Returns:
            Tuple[str, bool, Path, Path]: A tuple containing the resolved model path,
            a boolean indicating the use of LoRA, the result path, and the checkpoints path.
        """
        use_lora: bool = (
            Path(model_path_or_name).exists()
            and Path(model_path_or_name, "adapter_config.json").exists()
        )
        self.logger.debug(f"LoRA detected: {use_lora}")

        if (
            model_path_or_name != self.model_config.model_name
            and not Path(model_path_or_name).exists()
        ):
            self.logger.warning(
                f"Model path {model_path_or_name} not found. Using base model."
            )

            model_path_or_name = self.model_config.model_name
            use_lora = False

        if model_path_or_name == self.model_config.model_name:
            result_path, checkpoints_path = self._resolve_model_paths(False)
        else:
            model_dir: Path = Path(model_path_or_name).parent.parent

            if not model_dir.exists() or model_dir == Path():
                self.logger.warning(
                    f"Invalid model directory {model_dir}, using base model paths"
                )
                model_dir = self.config.base_path_config.models_dir / "base_model"

            result_path = model_dir / "result"
            checkpoints_path = model_dir / "checkpoints"

        return model_path_or_name, use_lora, result_path, checkpoints_path

    def _load_for_training(
        self,
        model_path_or_name: ModelPathOrName,
        torch_dtype: torch.dtype,
        use_lora: bool,
    ) -> InitializedModelType:
        """Loads and configures model for training scenarios.

        Args:
            model_path_or_name: The path to the model directory or model name, e.g. "facebook/nllb-200-distilled-1.3B".
            torch_dtype: The dtype of the model's tensors.
            use_lora: Whether to use LoRA adapters.

        Returns:
             PeftModel | PeftMixedModel | PreTrainedModel: The loaded model.
        """
        if use_lora:
            model_name = self.model_config.model_name
            self.logger.info(f"Loading training model (LoRA={use_lora}): {model_name}")

            model = self._load_base_model(model_name, torch_dtype)
            model = self._apply_lora(model)
        else:
            load_path = model_path_or_name
            self.logger.info(f"Loading training model (LoRA={use_lora}): {load_path}")

            if Path(load_path).exists():
                # Check if the path is a checkpoint
                if "checkpoint-" in load_path:
                    self.logger.info(f"Loading from checkpoint: {load_path}")

                model = self._load_base_model(load_path, torch_dtype)
            else:
                if load_path != self.model_config.model_name:
                    self.logger.warning(
                        f"Model path {load_path} not found, falling back to base model"
                    )

                load_path = self.model_config.model_name
                model = self._load_base_model(load_path, torch_dtype)

        return model

    def _load_for_translate(
        self,
        model_path_or_name: ModelPathOrName,
        torch_dtype: torch.dtype,
        use_lora: bool,
    ) -> PeftModel | PreTrainedModel:
        """Loads model for translation with optional LoRA weights.

        Args:
            model_path_or_name: The path to the model directory.
            torch_dtype: The dtype of the model's tensors.
            use_lora: Whether to use LoRA adapters.

        Returns:
            PeftModel | PreTrainedModel: The loaded model.
        """
        self.logger.info(
            f"Loading translation model (LoRA={use_lora}): {model_path_or_name}"
        )

        if use_lora:
            base_model = self._load_base_model(
                self.model_config.model_name, torch_dtype
            )

            return PeftModel.from_pretrained(base_model, model_path_or_name)

        return self._load_base_model(model_path_or_name, torch_dtype)

    def initialize(
        self,
        for_training: bool,
        torch_dtype: torch.dtype = torch.float16,
        model_cli_path: ModelCLIType = None,
        with_lora: bool = False,
    ) -> InitializedModelType:
        """Main initialization entry point for training/translation modes.

        Args:
            for_training (bool): Whether to initialize model for training.
            torch_dtype (torch.dtype, optional): The dtype of the model's tensors. Defaults to torch.float16.
            model_path (ModelCLIType): The path to the model directory. Defaults to None.
            with_lora (bool, optional): Whether to use LoRA adapters. Defaults to False.

        Returns:
            PeftModel | PeftMixedModel | PreTrainedModel: The loaded model.
        """
        try:
            # Resolve model path
            model_path_or_name: ModelPathOrName = (
                model_cli_path or self.model_config.model_name
            )

            # Configure paths and LoRA usage
            if for_training:
                use_lora = with_lora
                result_path, checkpoints_path = self._resolve_model_paths(
                    use_lora, model_path_or_name
                )
                self.model_config.last_checkpoint = (
                    self.checkpoint_utils.get_latest_checkpoint(checkpoints_path)
                )
                self.logger.debug(
                    f"Last checkpoint for training: {self.model_config.last_checkpoint}"
                )
                model = self._load_for_training(
                    model_path_or_name, torch_dtype, use_lora
                )
            else:
                model_path_or_name, use_lora, result_path, checkpoints_path = (
                    self._handle_output_path_rollback(model_path_or_name)
                )
                model = self._load_for_translate(
                    model_path_or_name, torch_dtype, use_lora
                )

            # Update config with resolved paths
            self.model_config.result_path = result_path
            self.model_config.checkpoints_path = checkpoints_path
            self.logger.debug(
                f"Resolved paths: result={result_path}, checkpoints={checkpoints_path}"
            )

            self.logger.info(f"Model successfully loaded on {self.device.upper()}")

            return model
        except Exception as e:
            error_msg = f"Model initialization failed: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
