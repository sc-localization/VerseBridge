import gc
import logging
import torch


class MemoryManager:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def clear(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        gc.collect()

        if torch.cuda.is_available():
            mem_before = (
                torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0
            )
            self.logger.debug(f"Memory before cleanup: {mem_before:.2f} MB")

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            mem_after = (
                torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0
            )

            self.logger.debug(f"Memory after cleanup: {mem_after:.2f} MB")
        else:
            self.logger.warning("⚠️ No GPU available, skipping CUDA memory cleanup")
