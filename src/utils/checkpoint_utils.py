import re
from pathlib import Path
from typing import Optional

from src.config.models import ModelConfig
from src.type_defs import ArgLoggerType, LastCheckpointPathType
from .app_logger import AppLogger


class CheckpointUtils:
    def __init__(
        self,
        logger: ArgLoggerType = None,
    ):
        self.logger = logger or AppLogger("checkpoint_utils").get_logger

    def get_latest_checkpoint(self, output_dir: Path) -> LastCheckpointPathType:
        """Returns the path to the latest checkpoint in the specified directory.

        Args:
            output_dir: Directory where checkpoints are searched for

        Returns:
            LastCheckpointPathType: Path to the latest checkpoint if it exists, otherwise None
        """
        if not output_dir.exists() or not output_dir.is_dir():
            self.logger.debug(f"No checkpoints found in {output_dir}")

            return None

        checkpoint_dirs = sorted(
            [
                d
                for d in output_dir.iterdir()
                if d.is_dir() and re.match(r"checkpoint-\d+", d.name)
            ],
            key=lambda x: int(x.name.split("-")[-1]),
        )

        latest = checkpoint_dirs[-1] if checkpoint_dirs else None

        if latest:
            self.logger.debug(f"Found latest checkpoint: {latest}")

        return latest

    def get_checkpoint_path(
        self,
        model_config: ModelConfig,
        checkpoint: Optional[str] = None,
    ) -> Path | None:
        """Returns the path to the checkpoint on the basis of the model configuration.

        Args:
            model_config: The configuration of the model.
            checkpoint: The name of the checkpoint or None to use the latest one.

        Returns:
            The path to the checkpoint or None if no checkpoint is found.
        """

        checkpoint_dir = model_config.checkpoints_path

        if checkpoint:
            final_checkpoint_dir = Path(checkpoint_dir) / checkpoint

            if not Path(final_checkpoint_dir).exists():
                self.logger.error(f"Checkpoint {final_checkpoint_dir} does not exist")
                raise ValueError(f"Checkpoint {final_checkpoint_dir} does not exist")

            self.logger.debug(f"Using specified checkpoint: {final_checkpoint_dir}")

            return final_checkpoint_dir

        final_checkpoint_dir = model_config.last_checkpoint

        if final_checkpoint_dir:
            self.logger.info(
                f"No checkpoint specified. Using latest checkpoint: {final_checkpoint_dir}"
            )
            return final_checkpoint_dir

        self.logger.info(f"No checkpoint found")

        return None
