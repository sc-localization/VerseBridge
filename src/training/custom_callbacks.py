import logging
from transformers import EarlyStoppingCallback, TrainerCallback

from src.utils import MemoryManager


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    """Early stopping callback with memory clearing."""

    def __init__(
        self,
        early_stopping_patience: int,
        early_stopping_threshold: float,
        logger: logging.Logger,
    ):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.logger = logger
        self.memory_manager = MemoryManager(self.logger)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore
        self.logger.debug("Performing evaluation. Clearing memory before evaluation...")
        self.memory_manager.clear()

        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)  # type: ignore

        self.logger.debug("Evaluation completed. Clearing memory after evaluation...")
        self.memory_manager.clear()


class LoggingCallback(TrainerCallback):
    """Callback for logging metrics and clearing memory."""

    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore
        logs = logs or {}
        formatted_logs = "\n".join(f"  {k}: {v}" for k, v in logs.items())
        self.logger.debug(
            f"Training step {state.global_step} metrics:\n{formatted_logs}"
        )

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore
        if state.global_step % 100 == 0:
            self.logger.debug(f"Step {state.global_step} completed. Clearing memory...")
