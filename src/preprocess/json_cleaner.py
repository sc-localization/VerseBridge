import html
import re
from typing import Dict, Set
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.utils import AppLogger
from src.type_defs import (
    ProtectedPatternsType,
    JSONDataListType,
    ArgLoggerType,
    INIFIleValueType,
    CleanedINIFIleValueType,
)


class JsonCleaner:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        protected_patterns: ProtectedPatternsType,
        max_model_length: int,
        logger: ArgLoggerType = None,
    ) -> None:
        """
        Initializes JsonCleaner with a tokenizer and configuration.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
            protected_patterns (ProtectedPatternsType): A list of patterns that should not be translated.
            max_model_length (int): The maximum length of tokens for the model.
            logger (ArgLoggerType, optional): A logger to use for logging operations (defaults to an AppLogger). Defaults to None.

        Returns:
            None
        """
        self.logger = logger or AppLogger("json_cleaner").get_logger

        self.tokenizer = tokenizer
        self.protected_patterns = protected_patterns
        self.max_model_length = max_model_length

        self.removed_count: Dict[str, int] = {
            "empty": 0,
            "duplicate": 0,
            "same_lang": 0,
            "too_long": 0,
            "too_short": 0,
            "foreign_words": 0,
            "length_mismatch": 0,
        }

    def clean_text(
        self, text: INIFIleValueType, remove_patterns: bool = True
    ) -> CleanedINIFIleValueType:
        """
        Transforms the text by removing special characters and improving readability.

        Args:
            text (INIFIleValueType): Input text.
            remove_patterns (bool): Flag to remove patterns.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        if remove_patterns:
            for pattern in self.protected_patterns:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove invisible symbols
        text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)

        # Translate \n
        text = text.replace("\\n", "\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r" {2,}\n", "\n", text)
        text = re.sub(r"\n {2,}", "\n", text)
        text = re.sub(r"\n", " ", text)

        # Decode HTML entities
        text = html.unescape(text)
        text = text.replace("\xa0", " ")

        # Normalize quotes and apostrophes
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("«", '"').replace("»", '"')

        # Handle ellipsis and punctuation
        text = text.replace("…", "...")
        text = re.sub(r"[,!?]{2,}", lambda m: m.group(0)[0], text)
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"[\u2013\u2014]", "-", text)
        text = re.sub(r"--", "-", text)
        text = re.sub(r"[-_=]{2,}", "", text)
        text = re.sub(r"\.{2,}", "...", text)

        def process_word_unit(word_unit: str) -> str:
            for pattern in self.protected_patterns:
                match = re.search(pattern, word_unit)

                if match:
                    found = match.group(0)
                    paren_match = re.fullmatch(r"(#*)?(~[a-zA-Z0-9]+\((.*?)\))", found)

                    if paren_match:
                        hashes = paren_match.group(1) or ""
                        full_func = paren_match.group(2)
                        inner = paren_match.group(3).strip()

                        if not inner:
                            return ""

                        if hashes == "#":
                            return hashes + full_func

                        return full_func

                    # Save the pattern and the rest of the word
                    start, end = match.span()

                    if start == 0:  # Pattern at the beginning of the word
                        remaining = word_unit[end:]

                        if re.search(r"[A-Za-zА-Яа-яЁё0-9]", remaining):
                            return found + remaining

                        return found
                    elif end == len(word_unit):  # Pattern at the end of the word
                        prefix = word_unit[:start]

                        if re.search(r"[A-Za-zА-Яа-яЁё0-9]", prefix):
                            return prefix + found

                        return found

                    return word_unit  # Pattern inside the word, save everything

            # TODO: use unicode for multi-language support
            if not re.search(r"[A-Za-zА-Яа-яЁё0-9]", word_unit):
                return ""

            m_trim = re.search(r"([\('\"]*[A-Za-zА-ЯЁёа-я].*)", word_unit)

            if m_trim:
                word_unit = m_trim.group(1)

            pattern = re.compile(
                r"^(.*[\(\)\'\"A-Za-zА-ЯЁёа-я])"
                r"(?:[^,.:;!?…A-Za-zА-ЯЁёа-я]*"
                r"(?:(?P<punc1>\.{3})|(?P<punc2>[,.:;!?])(?!\s)))?"
                r".*$"
            )

            m = pattern.match(word_unit)

            if m:
                word = m.group(1)
                punct = m.group("punc1") or m.group("punc2") or ""

                return word + punct

            return word_unit

        words = text.split()
        processed_words = [process_word_unit(word) for word in words]
        result_text = " ".join(word for word in processed_words if word)
        result_text = result_text.strip()

        return result_text

    def contains_foreign_words(
        self,
        cleaned_original_text: CleanedINIFIleValueType,
        cleaned_translated_text: CleanedINIFIleValueType,
    ) -> bool:
        """
        Checks if there are any untranslated words in the target text that are present in the original text.

        Args:
            cleaned_original_text (CleanedINIFIleValueType): Original text.
            cleaned_translated_text (CleanedINIFIleValueType): Translated text.

        Returns:
            True if there are untranslated words, False otherwise.
        """
        for pattern in self.protected_patterns:
            cleaned_translated_text = re.sub(pattern, "", cleaned_translated_text)
            cleaned_original_text = re.sub(pattern, "", cleaned_original_text)

        target_words = re.findall(r"\b\w+\b", cleaned_translated_text)
        cleaned_original_text = cleaned_original_text.strip().lower()

        for word_orig in target_words:
            target_word_cleaned = word_orig.strip().lower()

            if target_word_cleaned.isdigit():
                continue

            if target_word_cleaned in cleaned_original_text:
                return True

        return False

    def clean_json_data(self, data: JSONDataListType) -> JSONDataListType:
        """
        Cleans JSON data, removing duplicates, empty records, and incorrect texts.

        Args:
            data (JSONDataListType): A list of dictionaries with data.

        Returns:
            Cleaned data.
        """
        seen_pairs: Set[str] = set()
        cleaned_data: JSONDataListType = []

        for item in tqdm(data, desc="Cleaning data"):
            original_text: CleanedINIFIleValueType = self.clean_text(
                item.get("original", ""), remove_patterns=False
            ).lower()
            translated_text: CleanedINIFIleValueType = self.clean_text(
                item.get("translated", ""), remove_patterns=False
            ).lower()

            if not original_text or not translated_text:
                self.removed_count["empty"] += 1
                continue

            pair = f"{original_text}|||{translated_text}"
            if pair in seen_pairs:
                self.removed_count["duplicate"] += 1
                continue

            if original_text.lower() == translated_text.lower():
                self.removed_count["same_lang"] += 1
                continue

            original_tokens = self.tokenizer(
                original_text, add_special_tokens=True, padding=True, truncation=True
            )["input_ids"]
            translated_tokens = self.tokenizer(
                translated_text, add_special_tokens=True, padding=True, truncation=True
            )["input_ids"]
            original_tokens_len = len(original_tokens)  # type:ignore
            translated_tokens_len = len(translated_tokens)  # type:ignore

            if (
                original_tokens_len > self.max_model_length
                or translated_tokens_len > self.max_model_length
            ):
                self.removed_count["too_long"] += 1
                continue

            if len(original_text) < 2 or len(translated_text) < 2:
                self.removed_count["too_short"] += 1
                continue

            length_ratio = max(original_tokens_len, translated_tokens_len) / max(
                1, min(original_tokens_len, translated_tokens_len)
            )

            if length_ratio > 2.5:
                self.removed_count["length_mismatch"] += 1
                continue

            if self.contains_foreign_words(original_text, translated_text):
                self.removed_count["foreign_words"] += 1
                continue

            seen_pairs.add(pair)
            cleaned_data.append({"original": original_text, "translated": translated_text})

        self._log_cleaning_stats(len(cleaned_data))

        return cleaned_data

    def _log_cleaning_stats(self, final_count: int):
        self.logger.info(
            f"Removed: empty={self.removed_count['empty']}, "
            f"Duplicates={self.removed_count['duplicate']}, "
            f"Same language={self.removed_count['same_lang']}, "
            f"Too long={self.removed_count['too_long']}, "
            f"Too short={self.removed_count['too_short']}, "
            f"Foreign words={self.removed_count['foreign_words']}, "
            f"Length mismatch={self.removed_count['length_mismatch']}"
        )
        self.logger.info(f"Final dataset size: {final_count} pairs")
