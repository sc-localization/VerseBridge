from dataclasses import asdict, dataclass, field
from enum import Enum
from src.type_defs import (
    Optimizer,
    Scheduler,
    LogReportTarget,
    LogReportTargetListType,
    Metric,
    LabelNames,
    TranslationTrainingConfigType,
    Strategy,
    LabelNamesListType,
    NerTrainingConfigType,
)


@dataclass
class NerTrainingConfig:
    logging_dir: str

    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    eval_strategy: Strategy

    def to_dict(self) -> NerTrainingConfigType:
        return asdict(self)  # type: ignore


@dataclass
class TranslationTrainingConfig:
    """
    Parameters for training Hugging Face Transformers models.
    See:
    - https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    """

    logging_dir: str  # Log directory (initialized in __post_init__)

    # === General training parameters ===
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float

    # === Optimizer and scheduler ===
    optim: Optimizer
    lr_scheduler_type: Scheduler

    # === Batching and gradients ===
    bf16: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    eval_accumulation_steps: int

    # === Validation and saving strategies ===
    eval_strategy: Strategy
    eval_steps: int
    eval_on_start: bool
    predict_with_generate: bool
    save_strategy: Strategy
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: Metric
    greater_is_better: bool

    # === Logging and reporting ===
    logging_steps: int
    logging_strategy: Strategy
    report_to: LogReportTargetListType

    # === Label and generation ===
    label_smoothing_factor: float
    label_names: LabelNamesListType

    # === Additional parameters ===
    torch_empty_cache_steps: int
    group_by_length: bool
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    torch_compile: bool

    def to_dict(self) -> TranslationTrainingConfigType:
        config_as_dict = asdict(self)

        for key, value in config_as_dict.items():
            if isinstance(value, Enum):
                config_as_dict[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                config_as_dict[key] = [item.value for item in value]

        return config_as_dict  # type: ignore
