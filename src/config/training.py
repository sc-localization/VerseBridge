from dataclasses import asdict, dataclass, field
from enum import Enum

from src.type_defs import (
    Optimizer,
    Scheduler,
    LogReportTarget,
    LogReportTargetListType,
    Metric,
    LabelNames,
    TrainingConfigType,
    Strategy,
    LabelNamesListType,
)


@dataclass
class TrainingConfig:
    """
    Parameters for training Hugging Face Transformers models.
    See:
    - https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    """

    logging_dir: str  # Log directory (initialized in __post_init__)

    # === General training parameters ===
    num_train_epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # === Optimizer and scheduler ===
    optim: Optimizer = Optimizer.adafactor
    lr_scheduler_type: Scheduler = Scheduler.cosine

    # === Batching and gradients ===
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    eval_accumulation_steps: int = 4

    # === Validation and saving strategies ===
    eval_strategy: Strategy = Strategy.STEPS
    eval_steps: int = 200
    eval_on_start: bool = True
    predict_with_generate: bool = True
    save_strategy: Strategy = Strategy.STEPS
    save_steps: int = 200
    save_total_limit: int = 15
    load_best_model_at_end: bool = True
    metric_for_best_model: Metric = Metric.BERTSCORE_F1
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

    def to_dict(self) -> TrainingConfigType:
        config_as_dict = asdict(self)

        for key, value in config_as_dict.items():
            if isinstance(value, Enum):
                config_as_dict[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                config_as_dict[key] = [item.value for item in value]

        return config_as_dict  # type: ignore
