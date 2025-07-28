from pathlib import Path

from src.utils import AppLogger, FileUtils
from src.type_defs import (
    ArgLoggerType,
    ExcludeKeysType,
    INIDataType,
    LoggerType,
    JSONDataTranslationListType,
    INIFIleValueType,
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

    def create_json_data(
        self, original_ini: Path, translated_ini: Path
    ) -> JSONDataTranslationListType:
        """
        Creates a list of dictionaries with original-translated pairs from two INI files.

        Args:
            original_ini: Path - Path to the original INI file.
            translated_ini: Path - Path to the translated INI file.

        Returns:
            JSONDataListType: A list of dictionaries with {"original": str, "translated": str} pairs.
        """
        self.logger.debug(
            f"Creating JSON data from {original_ini} and {translated_ini}"
        )

        original_data: INIDataType = self.file_utils.parse_ini_file(
            original_ini, self.exclude_keys
        )
        translated_data: INIDataType = self.file_utils.parse_ini_file(
            translated_ini, self.exclude_keys
        )

        results: JSONDataTranslationListType = []

        for key in original_data:
            original_value: INIFIleValueType | None = original_data.get(key)
            translated_value: INIFIleValueType | None = translated_data.get(key)

            # key may be missing
            if original_value is None:
                self.logger.warning(
                    f"Key '{key}' not found in original file: {original_ini}"
                )
            if translated_value is None:
                self.logger.warning(
                    f"Key '{key}' not found in translated file: {translated_ini}"
                )

            if original_value and translated_value:
                results.append(
                    {"original": original_value, "translated": translated_value}
                )

        self.logger.debug(f"Created {len(results)} original-translated pairs")

        return results
