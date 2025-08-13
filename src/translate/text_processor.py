import nltk
import re
from nltk.tokenize import sent_tokenize
from typing import List, Optional, Tuple, Dict
from transformers import PreTrainedTokenizerBase

from src.type_defs import (
    ProtectedPatternsType,
    PlaceholdersType,
    TranslatorCallableType,
    INIFIleValueType,
    TranslatedIniValueType,
    LoggerType,
    TokenizerConfigType,
)

nltk.download("punkt_tab", quiet=True)


class TextProcessor:
    def __init__(
        self,
        protected_patterns: ProtectedPatternsType,
        logger: LoggerType,
    ):
        self.protected_pattern = re.compile("|".join(protected_patterns), re.UNICODE)
        self.logger = logger

    def _protect_patterns(
        self, text: INIFIleValueType, ner_patterns: Optional[List[str]] = None
    ) -> Tuple[INIFIleValueType, PlaceholdersType, PlaceholdersType]:
        """
        Protects both protected patterns and NER entities with unique placeholders.

        Args:
            text (str): Input text.
            ner_patterns (Optional[List[str]]): Regex patterns for NER entities.

        Returns:
            Tuple[str, Dict[str, str], Dict[str, str]]: Modified text, protected placeholders, NER placeholders.
        """
        protected_placeholders: PlaceholdersType = {}
        ner_placeholders: PlaceholdersType = {}
        modified_text = text

        # Step 1: Protect protected patterns
        if self.protected_pattern:

            def _replace_protected_match(match: re.Match[str]) -> str:
                match_value = match.group(0)
                key = f"[PP_{len(protected_placeholders)}]"
                protected_placeholders[key] = match_value
                self.logger.debug(f"Protected pattern: {match_value} -> {key}")
                return key

            try:
                modified_text = self.protected_pattern.sub(
                    _replace_protected_match, modified_text
                )
            except re.error as e:
                self.logger.error(f"Error applying protected pattern: {e}")
                raise

        # Step 2: Protect NER patterns
        if ner_patterns:
            try:
                ner_pattern = re.compile("|".join(ner_patterns), re.UNICODE)

                def _replace_ner_match(match: re.Match[str]) -> str:
                    match_value = match.group(0)
                    # Check if the match is already a placeholder
                    if re.match(r"\[PP_\d+\]", match_value):
                        self.logger.debug(
                            f"Skipping NER match as it is a protected placeholder: {match_value}"
                        )

                        return match_value

                    key = f"[NER_{len(ner_placeholders)}]"
                    ner_placeholders[key] = match_value

                    return key

                modified_text = ner_pattern.sub(_replace_ner_match, modified_text)
            except re.error as e:
                self.logger.error(f"Error compiling NER pattern: {e}")
                raise

        # Step 3: Protect newlines
        modified_text = modified_text.replace("\\n", "[NL]")

        return modified_text, protected_placeholders, ner_placeholders

    def _restore_patterns(
        self,
        translated_text: TranslatedIniValueType,
        protected_placeholders: PlaceholdersType,
        ner_placeholders: PlaceholdersType,
        ner_translations: Optional[Dict[str, str]] = None,
    ) -> TranslatedIniValueType:
        """
        Restores protected and NER placeholders. NER can be restored as originals or translated.

        Args:
            translated_text (str): Translated text with placeholders.
            protected_placeholders (Dict[str, str]): Protected pattern placeholders.
            ner_placeholders (Dict[str, str]): NER placeholders.
            ner_translations (Optional[Dict[str, str]]): Translations for NER (None for context version).

        Returns:
            str: Text with placeholders restored.
        """
        result = translated_text

        # Step 1: Restore NER placeholders (to handle potential overlaps first)
        for key, original in ner_placeholders.items():
            value = (
                ner_translations.get(original, original)
                if ner_translations
                else original
            )

            result = re.sub(re.escape(key), value, result)

        # Step 2: Restore protected placeholders
        for key, value in protected_placeholders.items():
            result = re.sub(re.escape(key), value, result)

        # Step 3: Restore newlines
        result = result.replace("[NL]", "\\n")

        return result

    def split_text(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        tokenizer_args: TokenizerConfigType,
        depth: int = 0,
        max_depth: int = 5,
    ) -> List[str]:
        """
        Splits a given text into chunks that will not exceed the maximum number of tokens.

        Args:
            text (str): Input text.
            tokenizer (PreTrainedTokenizerBase): A tokenizer to use for counting tokens.
            max_tokens (int): The maximum number of tokens in a chunk.

        Returns:
            List[str]: A list of chunks.
        """

        if depth > max_depth:
            self.logger.warning(
                f"Max split depth {max_depth} reached. Truncating text."
            )

            tokens = tokenizer(text, **tokenizer_args)

            return [
                tokenizer.decode(
                    tokens["input_ids"][0][:max_tokens],  # type:ignore
                    skip_special_tokens=True,
                )
            ]

        sentences: List[str] = sent_tokenize(text)
        chunks: List[str] = []
        current_chunk: str = ""

        for sentence in sentences:
            prospective_chunk = (
                f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            )
            tokens = tokenizer(prospective_chunk, **tokenizer_args)
            token_len = len(tokens["input_ids"][0])  # type:ignore

            if token_len > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # sentence too long → try recursively splitting it
                sub_chunks = self.split_text(
                    sentence,
                    tokenizer,
                    max_tokens,
                    tokenizer_args,
                    depth + 1,
                    max_depth,
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = prospective_chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def translate_text(
        self,
        text: INIFIleValueType,
        translator: TranslatorCallableType,
        ner_patterns: Optional[List[str]] = None,
        ner_cache: Optional[Dict[str, str]] = None,
    ) -> Tuple[TranslatedIniValueType, TranslatedIniValueType]:
        """
        Translates text, returning both context (NER protected) and full (NER translated) versions.

        Args:
            text (str): Input text.
            translator (TranslatorCallableType): Translator function.
            ner_patterns (Optional[List[str]]): Patterns for NER protection.
            ner_cache (Optional[Dict[str, str]]): Cache for NER translations.

        Returns:
            Tuple[str, str]: (context_version, full_version)
        """
        if not text:
            return text, text

        # Step 1: Protect both patterns
        modified_text, protected_placeholders, ner_placeholders = (
            self._protect_patterns(text, ner_patterns)
        )

        # Step 2: Translate the modified text (once)
        translated_text = translator(modified_text)

        # Step 3: Restore for context (NER unchanged)
        context_version = self._restore_patterns(
            translated_text, protected_placeholders, ner_placeholders
        )

        # Step 4: Restore for full (translate NER)
        ner_translations: Dict[str, str] = ner_cache or {}

        for original in ner_placeholders.values():
            if original not in ner_translations:
                translated_ner = translator(original)
                ner_translations[original] = translated_ner

        full_version = self._restore_patterns(
            translated_text, protected_placeholders, ner_placeholders, ner_translations
        )

        return context_version, full_version
