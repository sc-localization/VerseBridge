from pathlib import Path
from typing import Set, Tuple, List, TYPE_CHECKING, cast

from tqdm import tqdm

from src.config import ConfigManager
from src.utils import FileUtils, NerUtils
from src.type_defs import (
    ExcludeKeysType,
    INIDataType,
    LoggerType,
    INIFIleKeyType,
    INIFIleValueType,
    TranslatedIniLineType,
    PlaceholdersType,
    is_list_of_ner_patterns,
)
from .text_processor import TextProcessor
from .buffered_file_writer import BufferedFileWriter

if TYPE_CHECKING:
    from .translator import Translator


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
        if key in exclude_keys or not value:
            return False
        return key in missing_keys

    def _read_files(
        self,
        translation_input_file: Path,
        context_file: Path,
        full_file: Path,
    ) -> Tuple[INIDataType, INIDataType]:
        try:
            input_items: INIDataType = self.file_utils.parse_ini_file(
                translation_input_file
            )
            self.logger.debug(
                f"Read {len(input_items)} key-value pairs from {translation_input_file}"
            )
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
        input_keys: Set[str] = set(input_items.keys())
        result_keys: Set[str] = set(result_items.keys())
        exclude_keys: ExcludeKeysType = self.config.translation_config.exclude_keys

        missing_keys = input_keys - result_keys
        obsolete_keys = result_keys - input_keys
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
        translator: "Translator",
    ) -> None:
        self.logger.info(f"Translating file: {translation_input_file}")
        self.logger.debug(f"Base translation file: {translation_result_file}")

        translation_dest_dir = translation_result_file.parent

        if not translation_dest_dir.exists():
            raise FileNotFoundError(
                f"Output directory {translation_dest_dir} does not exist"
            )

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

        ner_patterns_path = self.config.ner_path_config.ner_patterns_path

        if ner_patterns_path.exists():
            ner_patterns = self.file_utils.load_json(ner_patterns_path)
        else:
            ner_patterns = self.ner_utils.get_ner_patterns(
                self.config.ner_path_config.corrected_streamlit_data_path,
                self.config.ner_path_config.extracted_ner_data_path,
            )
            self.file_utils.save_json(ner_patterns, ner_patterns_path)

        if not is_list_of_ner_patterns(ner_patterns):
            raise TypeError("NER patterns must be a list of strings")

        keys_to_translate = sorted(list(missing_keys))
        buffer_size = self.config.translation_config.buffer_size

        with (
            BufferedFileWriter(
                context_file, buffer_size, self.logger
            ) as context_writer,
            BufferedFileWriter(full_file, buffer_size, self.logger) as full_writer,
        ):
            # 1. First, we write down ALL existing (old) translations
            existing_keys = sorted(
                [k for k in result_items.keys() if k not in obsolete_keys]
            )

            for key in existing_keys:
                val = result_items[key]
                # Assume that the old values ​​have already been translated (strings)
                context_writer.write(cast(TranslatedIniLineType, f"{key}={val}\n"))
                full_writer.write(cast(TranslatedIniLineType, f"{key}={val}\n"))

            # Force flushing the buffer of old lines before starting new ones
            context_writer.flush()
            full_writer.flush()

            # 2. Process NEW keys in batches of buffer_size
            total_new = len(keys_to_translate)
            self.logger.info(
                f"Translating {total_new} new keys in batches of {buffer_size}..."
            )

            for i in tqdm(range(0, total_new, buffer_size), desc="File Progress"):
                chunk_keys = keys_to_translate[i : i + buffer_size]

                batch_context_inputs: List[str] = []
                batch_full_inputs: List[str] = []
                restoration_data: List[Tuple[PlaceholdersType, PlaceholdersType]] = []

                for key in chunk_keys:
                    text = input_items[key]
                    context_text, full_text, prot_ph, ner_ph = (
                        self.text_processor.protect_patterns(text, ner_patterns)
                    )
                    batch_context_inputs.append(context_text)
                    batch_full_inputs.append(full_text)
                    restoration_data.append((prot_ph, ner_ph))

                # Translate this chunk
                total_inputs = batch_context_inputs + batch_full_inputs

                batch_size = self.config.translation_config.batch_size
                all_translated_results = translator.translate_batch(
                    total_inputs, batch_size
                )

                mid = len(batch_context_inputs)
                trans_ctx = all_translated_results[:mid]
                trans_full = all_translated_results[mid:]

                for idx, key in enumerate(chunk_keys):
                    prot_ph, ner_ph = restoration_data[idx]

                    fin_ctx = self.text_processor.restore_patterns(
                        trans_ctx[idx], prot_ph, ner_ph
                    )
                    fin_full = self.text_processor.restore_patterns(
                        trans_full[idx], prot_ph, {}
                    )

                    context_writer.write(
                        cast(TranslatedIniLineType, f"{key}={fin_ctx}\n")
                    )
                    full_writer.write(
                        cast(TranslatedIniLineType, f"{key}={fin_full}\n")
                    )

                # Save to disk after each chunk
                context_writer.flush()
                full_writer.flush()
