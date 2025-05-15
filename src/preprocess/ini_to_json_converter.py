from typing import Optional
from pathlib import Path

from src.utils import AppLogger, FileUtils
from src.type_defs import (
    ArgLoggerType,
    ExcludeKeysType,
    INIDataType,
    LoggerType,
    JSONDataListType,
)


class IniConverter:
    def __init__(
        self,
        exclude_keys: ExcludeKeysType,
        logger: ArgLoggerType = None,
    ) -> None:
        """
        Initializes IniConverter with a set of keys to exclude from parsing.

        Args:
            exclude_keys: ExcludeKeysType - A set of keys to exclude from parsing.
            logger: ArgLoggerType - A logger for logging operations (default is created through AppLogger).
        """
        self.logger: LoggerType = logger or AppLogger("ini_converter").get_logger
        self.exclude_keys: ExcludeKeysType = exclude_keys
        self.file_utils: FileUtils = FileUtils(self.logger)

    def create_json_data(self, source_ini: Path, target_ini: Path) -> JSONDataListType:
        """
        Creates a list of dictionaries with source-target pairs from two INI files.

        Args:
            source_ini: Path - Path to the source INI file.
            target_ini: Path - Path to the target INI file.

        Returns:
            JSONDataListType: A list of dictionaries with {"source": str, "target": str} pairs.
        """
        self.logger.debug(f"Creating JSON data from {source_ini} and {target_ini}")

        source_data: INIDataType = self.file_utils.parse_ini_file(
            source_ini, self.exclude_keys
        )
        target_data: INIDataType = self.file_utils.parse_ini_file(
            target_ini, self.exclude_keys
        )

        results: JSONDataListType = []

        for key in source_data:
            source_value: Optional[str] = source_data[key]
            target_value: Optional[str] = target_data[key]

            if source_value and target_value:
                results.append({"source": source_value, "target": target_value})

        self.logger.debug(f"Created {len(results)} source-target pairs")

        return results
