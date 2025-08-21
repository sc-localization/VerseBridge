import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from typing import List, Set
from pathlib import Path


from src.utils import FileUtils
from src.type_defs import LoggerType, is_json_data_ner_list_type

nltk.download("stopwords", quiet=True)


class NerUtils:
    def __init__(self, logger: LoggerType):
        self.logger = logger

    def _extract_ner_entities(
        self,
        corrected_file_path: Path,
        unannotated_file_path: Path,
    ) -> Set[str]:
        """
        Extracts unique NER entities (text spans) from a corrected or unannotated JSON file.

        Args:
            corrected_file_path (Path): Path to dataset_corrected.json.
            unannotated_file_path (Path): Path to ner_unannotated.json (fallback).

        Returns:
            Set[str]: Set of unique NER entities (actual text values).
        """
        file_utils = FileUtils(logger=self.logger)
        entities: Set[str] = set()

        selected_file = corrected_file_path

        if not corrected_file_path.exists():
            self.logger.warning(
                f"File {corrected_file_path} not found, using fallback: {unannotated_file_path}"
            )

            selected_file = unannotated_file_path

        if not selected_file.exists():
            self.logger.error(
                f"File {selected_file} not found. NER entities will not be extracted."
            )

            return entities

        try:
            ner_data = file_utils.load_json(selected_file)

            if not is_json_data_ner_list_type(ner_data):
                raise ValueError("Invalid NER data format")

            if not ner_data:
                self.logger.warning(
                    f"File {selected_file} is empty. NER entities not extracted."
                )

                return entities

            for item in tqdm(ner_data, desc="Extracting NER entities"):
                text = item.get("text", "")

                if not text:
                    self.logger.warning(
                        f"Empty text in record {item.get('id', 'unknown')} in {selected_file}"
                    )
                    continue

                entities_list = item.get("entities", [])

                if not entities_list:
                    self.logger.debug(
                        f"Empty entities list in record {item.get('id', 'unknown')} in {selected_file}"
                    )
                    continue

                for entity in entities_list:
                    start = entity.get("start")
                    end = entity.get("end")
                    label = entity.get("label")

                    if 0 <= start <= end <= len(text):
                        entity_text = text[start:end].strip()
                        # Cleaning from punctuation at the beginning and end, as well as an apostrophe,
                        # if the named entity contains them (incorrect dataset annotation)
                        # UEE's -> UEE
                        cleaned_entity_text = (
                            re.sub(r"^[^\w\s]+|[^\w\s]+$", "", entity_text)
                            .replace("'", "")
                            .strip()
                        )

                        if cleaned_entity_text.lower() in stopwords.words("english"):
                            self.logger.debug(f"Skipping excluded word: {entity_text}")
                            continue

                        if cleaned_entity_text not in entities:
                            entities.add(cleaned_entity_text)
                        elif not cleaned_entity_text:
                            self.logger.warning(
                                f"Empty entity for start={start}, end={end}, label={label} in {selected_file}"
                            )
                    else:
                        self.logger.warning(
                            f"Invalid entity format {entity} in {selected_file}"
                        )

            self.logger.info(
                f"Extracted {len(entities)} unique NER entities from {selected_file}"
            )

        except Exception as e:
            self.logger.error(f"Error extracting NER entities: {e}")

        return entities

    def get_ner_patterns(
        self, corrected_file_path: Path, unannotated_file_path: Path
    ) -> List[str]:
        """
        Generates regex patterns for NER entities.

        Args:
            corrected_file_path (Path): Path to corrected NER data.
            unannotated_file_path (Path): Path to unannotated NER data (fallback).

        Returns:
            List[str]: List of escaped regex patterns for NER entities.
        """

        entities = self._extract_ner_entities(
            corrected_file_path, unannotated_file_path
        )
        patterns: List[str] = []

        for entity in sorted(entities, key=len, reverse=True):
            escaped = re.escape(entity)
            # Add word boundaries, but only if the entity does not contain tags or special symbols
            if not any(c in entity for c in "<>[]{}%~"):
                pattern = r"\b" + escaped + r"\b"
            else:
                pattern = escaped

            patterns.append(pattern)

        self.logger.info(f"Generated {len(patterns)} NER patterns")

        return patterns
