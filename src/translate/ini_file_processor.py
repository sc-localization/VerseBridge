from pathlib import Path
from tqdm import tqdm
from typing import Set, Tuple


from src.config import ConfigManager
from src.utils import FileUtils, NerUtils
from src.type_defs import (
    ExcludeKeysType,
    INIDataType,
    TranslatorCallableType,
    LoggerType,
    INIFIleKeyType,
    INIFIleValueType,
    TranslatedIniLineType,
    is_ini_file_line,
    is_list_of_ner_patterns,
    JSONNERListType,
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
        self.ner_utils = NerUtils(logger=logger)

    def _should_translate(
        self,
        key: INIFIleKeyType,
        value: INIFIleValueType,
        exclude_keys: ExcludeKeysType,
        missing_keys: Set[INIFIleKeyType],
    ) -> bool:
        """
        Checks if a key-value pair should be translated.

        Args:
            key (str): The key to check.
            value (str): The value to check.
            exclude_keys (ExcludeKeysType): The keys to exclude from translation.
            missing_keys (Set[str]): Keys present in input but not in output.

        Returns:
            bool: True if the key-value pair should be translated, False otherwise.
        """
        if key in exclude_keys or not value:
            return False

        return key in missing_keys

    def _process_line(
        self,
        key: INIFIleKeyType,
        value: INIFIleValueType,
        translator: TranslatorCallableType,
        processed_lines: INIDataType,
        exclude_keys: ExcludeKeysType,
        missing_keys: Set[INIFIleKeyType],
        ner_patterns: JSONNERListType,
    ) -> Tuple[TranslatedIniLineType, TranslatedIniLineType]:
        """
        Processes a line, returning both context and full versions.

        Args:
            key (str): The key of the line.
            value (str): The value of the line.
            translator (TranslatorCallableType): A callable for translating the text.
            processed_lines (INIDataType): A dictionary containing lines to be processed (translated or existing).
            exclude_keys (ExcludeKeysType): A tuple of keys to exclude from translation.
            missing_keys (Set[str]): Keys present in input but not in result.
            ner_patterns (JSONNERListType): Patterns for NER protection.

        Returns:
            Tuple[str, str]: The processed line in the format 'key=value'.
        """

        if self._should_translate(key, value, exclude_keys, missing_keys):
            context_value, full_value = self.text_processor.translate_text(
                value, translator, ner_patterns
            )

        else:
            context_value = full_value = processed_lines.get(key, value)

        context_line = f"{key}={context_value}\n"
        full_line = f"{key}={full_value}\n"

        if not is_ini_file_line(context_line) or not is_ini_file_line(full_line):
            raise ValueError(f"Invalid INI line format for key '{key}'")

        return context_line, full_line

    def _read_files(
        self,
        translation_input_file: Path,
        context_file: Path,
        full_file: Path,
    ) -> Tuple[INIDataType, INIDataType]:
        """
        Reads key-value pairs from input and existing result files (context or full).

        Args:
            translation_input_file (Path): Path to the input INI file.
            context_file (Path): Path to the context INI file.
            full_file (Path): Path to the full INI file.

        Returns:
            Tuple[INIDataType, INIDataType]: The source key-value pairs (input_items).
        """
        try:
            input_items: INIDataType = self.file_utils.parse_ini_file(
                translation_input_file
            )
            self.logger.debug(
                f"Read {len(input_items)} key-value pairs from {translation_input_file}"
            )
            # Load result_items from context or full if they exist
            result_items: INIDataType = {}

            if context_file.exists():
                result_items = self.file_utils.parse_ini_file(context_file)
                self.logger.debug(
                    f"Using context file for continuation: {len(result_items)} pairs from {context_file}"
                )
            elif full_file.exists():
                result_items = self.file_utils.parse_ini_file(full_file)
                self.logger.debug(
                    f"Using full file for continuation: {len(result_items)} pairs from {full_file}"
                )

            return input_items, result_items
        except Exception as e:
            self.logger.error(f"Failed to read files: {str(e)}")
            raise

    def _compare_keys(
        self,
        input_items: INIDataType,
        result_items: INIDataType,
    ) -> Tuple[Set[INIFIleKeyType], Set[INIFIleKeyType]]:
        """
        Compares keys of input and result dictionaries:
            1. missing_keys: Keys present in input_items but not in result_items.
            2. obsolete_keys: Keys present in result_items but not in input_items.

        Args:
            input_items: The dictionary of input items.
            result_items: The dictionary of result items from context/full file.

        Returns:
            Tuple[Set[str], Set[str]]: A tuple containing:
                missing_keys: Keys to be translated.
                obsolete_keys: Keys to remove.
        """
        input_keys: Set[str] = set(input_items.keys())
        result_keys: Set[str] = set(result_items.keys())
        exclude_keys: ExcludeKeysType = self.config.translation_config.exclude_keys

        # New keys: absent in result_items
        missing_keys = input_keys - result_keys
        # Obsolete keys: present in result_items but absent in input_items
        obsolete_keys = result_keys - input_keys
        # Filter out missing_keys, excluding exclude_keys and empty values
        missing_keys = {
            key for key in missing_keys if key not in exclude_keys and input_items[key]
        }

        self.logger.info(f"Found {len(missing_keys)} new keys for translation")
        self.logger.info(f"Found {len(obsolete_keys)} obsolete keys to remove")

        return missing_keys, obsolete_keys

    def translate_file(
        self,
        translation_input_file: Path,
        translation_result_file: Path,
        translator: TranslatorCallableType,
    ) -> None:
        """
        Translates an INI file, preserving existing translations if provided.

        Args:
            translation_input_file: Path to the input INI file.
            translation_result_file: Path to the translation result INI file.
            translator: A callable that takes a source and target language code
                and returns a translated string.
        """
        self.logger.info(f"Translating file: {translation_input_file}")
        self.logger.debug(f"Base translation file: {translation_result_file}")

        translation_dest_dir = translation_result_file.parent

        if not translation_dest_dir.exists():
            self.logger.error(f"Output directory {translation_dest_dir} does not exist")
            raise FileNotFoundError(
                f"Output directory {translation_dest_dir} does not exist"
            )

        # Create two files
        context_file = translation_result_file.with_name(
            translation_result_file.stem + "_context.ini"
        )
        full_file = translation_result_file.with_name(
            translation_result_file.stem + "_full.ini"
        )

        input_items, result_items = self._read_files(
            translation_input_file, context_file, full_file
        )

        missing_keys, obsolete_keys = self._compare_keys(input_items, result_items)

        if not missing_keys and not obsolete_keys:
            self.logger.info(f"No changes detected, skipping translation")
            return

        input_keys = set(input_items.keys())

        # Combine translations: result_items (from context/full) + new from input
        processed_lines: INIDataType = result_items.copy()

        for key in input_keys:
            if key not in processed_lines:
                processed_lines[key] = input_items[key]

        ner_patterns_path = self.config.ner_path_config.ner_patterns_path

        if ner_patterns_path.exists():
            self.logger.info(f"Loading cached NER patterns from {ner_patterns_path}")
            ner_patterns = self.file_utils.load_json(ner_patterns_path)
        else:
            self.logger.info("Extracting NER patterns (first run)")
            ner_patterns = self.ner_utils.get_ner_patterns(
                self.config.ner_path_config.corrected_streamlit_data_path,
                self.config.ner_path_config.extracted_ner_data_path,
            )

            self.file_utils.save_json(ner_patterns, ner_patterns_path)
            self.logger.info(
                f"Saved NER patterns to {ner_patterns_path} for future use"
            )

        if not is_list_of_ner_patterns(ner_patterns):
            raise TypeError("NER patterns must be a list of strings")

        with (
            BufferedFileWriter(
                context_file, self.config.translation_config.buffer_size, self.logger
            ) as context_writer,
            BufferedFileWriter(
                full_file, self.config.translation_config.buffer_size, self.logger
            ) as full_writer,
        ):
            for key, value in tqdm(
                sorted(processed_lines.items()),
                desc="Translating INI with NER split",
                total=len(processed_lines),
            ):
                if key in obsolete_keys:
                    continue

                context_line, full_line = self._process_line(
                    key,
                    value,
                    translator,
                    processed_lines,
                    self.config.translation_config.exclude_keys,
                    missing_keys,
                    ner_patterns,
                )

                context_writer.write(context_line)
                full_writer.write(full_line)
