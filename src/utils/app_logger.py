import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config import ConfigManager


class AppLogger:

    def __init__(
        self,
        name: Optional[str] = None,
        log_file: Optional[str] = None,
        log_dir: str | Path = "logs",
        file_level: int = logging.DEBUG,
        console_level: int = logging.INFO,
        backup_count: int = 7,
    ):
        """
        Initializes an AppLogger with the given parameters.

        Args:
            name: The name of the logger.
            log_dir: The directory for storing logs.
            log_file: The name of the log file.
            file_level: The logging level for the file (default is logging.DEBUG).
            console_level: The logging level for the console (default is logging.INFO).
        """

        config = ConfigManager()
        default_log_file, default_log_name = config.logging_config.default

        self.logger_name = name or default_log_name
        self.log_dir = Path(log_dir or config.path_config.logging_dir)
        self.log_file = self.log_dir / (log_file or default_log_file)

        if not self.log_file.name.endswith(".log"):
            raise ValueError("Log file must have .log extension")

        self.file_level = file_level
        self.console_level = console_level
        self.backup_count = backup_count

        # Создание логгера
        self._logger = logging.getLogger(self.logger_name)
        self._logger.setLevel(logging.DEBUG)  # Базовый уровень для логгера

        # Проверка, чтобы избежать дублирования обработчиков
        if not self._logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self) -> None:
        if self._logger.handlers:
            self._logger.handlers.clear()

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Handler for console (INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(self._get_formatter())

        # File handler (DEBUG and above)
        file_handler = TimedRotatingFileHandler(
            self.log_file,
            when="midnight",
            interval=1,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(self._get_formatter())

        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)

    def _get_formatter(self) -> logging.Formatter:
        return logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @property
    def get_logger(self) -> logging.Logger:
        return self._logger
