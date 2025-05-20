import os
from pathlib import Path
from tqdm import tqdm
from typing import Set, Tuple


from src.config import ConfigManager
from src.utils import FileUtils
from src.type_defs import (
    ExcludeKeysType,
    INIDataType,
    TranslatorCallableType,
    LoggerType,
)
from .text_processor import TextProcessor
from .buffered_file_writer import BufferedFileWriter


class IniFileProcessor:
    def __init__(
        self,
        config: ConfigManager,
        text_processor: TextProcessor,
        logger: LoggerType,
    ):
        self.config = config
        self.logger = logger
        self.text_processor = text_processor
        self.file_utils = FileUtils(logger=logger)

    def _should_translate(
        self,
        key: str,
        value: str,
        translated_lines: INIDataType,
        exclude_keys: ExcludeKeysType,
    ) -> bool:
        """
        Checks if a key-value pair should be translated.

        Args:
            key (str): The key to check.
            value (str): The value to check.
            translated_lines (INIDataType): The translated lines.
            exclude_keys (ExcludeKeysType): The keys to exclude from translation.

        Returns:
            bool: True if the key-value pair should be translated, False otherwise.
        """
        return key not in translated_lines and key not in exclude_keys and bool(value)

    def process_line(
        self,
        key: str,
        value: str,
        translator: TranslatorCallableType,
        translated_lines: INIDataType,
        exclude_keys: ExcludeKeysType,
    ) -> str:
        """
        Processes a line by translating its value if necessary.

        Args:
            key (str): The key of the line.
            value (str): The value of the line.
            translator (TranslatorCallableType): A callable for translating the text.
            translated_lines (INIDataType): A dictionary containing already translated lines.
            exclude_keys (ExcludeKeysType): A tuple of keys to exclude from translation.

        Returns:
            str: The processed line in the format 'key=value'.
        """
        if self._should_translate(key, value, translated_lines, exclude_keys):
            translated_value: str = self.text_processor.translate_text(
                value, translator
            )
        else:
            translated_value: str = translated_lines.get(key, value)

        translated_lines[key] = translated_value

        return f"{key}={translated_value}\n"

    def _read_files(
        self, input_file: Path, output_file: Path
    ) -> Tuple[INIDataType, INIDataType]:
        """
        Reads key-value pairs from input and output INI files.

        Args:
            input_file (Path): Path to the input INI file.
            output_file (Path): Path to the output INI file.

        Returns:
            Tuple[INIDataType, INIDataType]: A tuple containing the source and destination key-value pairs.
        """
        try:
            input_items: INIDataType = self.file_utils.parse_ini_file(input_file)

            self.logger.debug(
                f"Read {len(input_items)} key-value pairs from {input_file}"
            )

            output_items: INIDataType = {}

            if os.path.exists(output_file):
                output_items = self.file_utils.parse_ini_file(output_file)

            self.logger.debug(
                f"Read {len(output_items)} key-value pairs from {output_file}"
            )


            print(f"input items {input_items} output items {output_items}")
            return input_items, output_items
        except Exception as e:
            self.logger.error(f"Failed to read files: {str(e)}")
            raise

    def _compare_keys(
        self,
        input_items: INIDataType,
        output_items: INIDataType,
    ) -> Tuple[Set[str], Set[str]]:
        """
        Compares the keys of two dictionaries and returns two sets:
        1. missing_keys: The set of keys present in input_items but not in output_items.
        2. obsolete_keys: The set of keys present in output_items but not in input_items.

        Args:
            input_items (INIDataType): The dictionary of input items.
            output_items (INIDataType): The dictionary of output items.

        Returns:
            Tuple[Set[str], Set[str]]: A tuple containing the missing and obsolete keys.
        """
        input_keys: Set[str] = set(input_items.keys())
        output_keys: Set[str] = set(output_items.keys())

        missing_keys: Set[str] = input_keys - output_keys
        obsolete_keys: Set[str] = output_keys - input_keys

        self.logger.info(f"Found {len(missing_keys)} new keys for translation")
        self.logger.info(f"Found {len(obsolete_keys)} obsolete keys to remove")

        return missing_keys, obsolete_keys

    def translate_file(
        self,
        input_translation_file: Path,
        output_translation_file: Path,
        translator: TranslatorCallableType,
    ) -> None:
        self.logger.info(f"Translating file: {input_translation_file}")

        input_items, output_items = self._read_files(
            input_translation_file, output_translation_file
        )
        _, obsolete_keys = self._compare_keys(input_items, output_items)

        with BufferedFileWriter(
            output_translation_file,
            self.config.translation_config.buffer_size,
            self.logger,
        ) as writer:
            for key, value in tqdm(
                input_items.items(), desc="Translating INI", total=len(input_items)
            ):
                if key in obsolete_keys:
                    continue

                translated_line = self.process_line(
                    key,
                    value,
                    translator,
                    output_items,
                    self.config.translation_config.exclude_keys,
                )

                writer.write(translated_line)
