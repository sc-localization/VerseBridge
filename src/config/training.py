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

    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    eval_strategy: Strategy = Strategy.STEPS

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
    num_train_epochs: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # === Optimizer and scheduler ===
    optim: Optimizer = Optimizer.paged_adamw_8bit
    lr_scheduler_type: Scheduler = Scheduler.linear

    # === Batching and gradients ===
    fp16: bool = True
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    eval_accumulation_steps: int = 4

    # === Validation and saving strategies ===
    eval_strategy: Strategy = Strategy.STEPS
    eval_steps: int = 500
    eval_on_start: bool = False
    predict_with_generate: bool = True
    save_strategy: Strategy = Strategy.STEPS
    save_steps: int = 500
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: Metric = Metric.BLEU
    greater_is_better: bool = True

    # === Logging and reporting ===
    logging_steps: int = 50
    logging_strategy: Strategy = Strategy.STEPS
    report_to: LogReportTargetListType = field(
        default_factory=lambda: [LogReportTarget.TENSORBOARD]
    )  # Where to log

    # === Label and generation ===
    label_smoothing_factor: float = 0.05
    label_names: LabelNamesListType = field(default_factory=lambda: [LabelNames.LABELS])

    # === Additional parameters ===
    torch_empty_cache_steps: int = 100

    def to_dict(self) -> TranslationTrainingConfigType:
        config_as_dict = asdict(self)

        for key, value in config_as_dict.items():
            if isinstance(value, Enum):
                config_as_dict[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                config_as_dict[key] = [item.value for item in value]

        return config_as_dict  # type: ignore
