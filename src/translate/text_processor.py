import nltk
import re
from nltk.tokenize import sent_tokenize
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizerBase

from src.config import ConfigManager
from src.type_defs import (
    PlaceholdersType,
    TranslatorCallableType,
    INIFIleValueType,
    TranslatedIniValueType,
    LoggerType,
    TokenizerConfigType,
    JSONNERListType,
)

nltk.download("punkt_tab", quiet=True)


class TextProcessor:
    def __init__(
        self,
        config: ConfigManager,
        logger: LoggerType,
    ):
        self.config = config
        self.logger = logger
        self.translation_config = self.config.translation_config
        protected_patterns = self.translation_config.protected_patterns

        self.protected_pattern = re.compile("|".join(protected_patterns), re.UNICODE)

    def _protect_patterns(
        self, text: INIFIleValueType, ner_patterns: Optional[JSONNERListType] = None
    ) -> Tuple[INIFIleValueType, INIFIleValueType, PlaceholdersType, PlaceholdersType]:
        """
        Protects both protected patterns and NER entities with unique placeholders.

        Args:
            text (INIFIleValueType): Input text.
            ner_patterns (JSONNERListType): Regex patterns for NER entities.

        Returns:
            Tuple[INIFIleValueType, INIFIleValueType, Dict[str, str], Dict[str, str]]: Modified text, protected placeholders, NER placeholders.
        """
        protected_placeholders: PlaceholdersType = {}
        ner_placeholders: PlaceholdersType = {}

        context_text = text
        full_text = text

        # Step 1: Protect protected patterns
        if self.protected_pattern:
            try:

                def _replace_protected_match(match: re.Match[str]) -> str:
                    match_value = match.group(0)
                    key = self.translation_config.get_p_template(
                        len(protected_placeholders)
                    )
                    protected_placeholders[key] = match_value

                    return key

                context_text = self.protected_pattern.sub(
                    _replace_protected_match, context_text
                )
                full_text = self.protected_pattern.sub(
                    _replace_protected_match, full_text
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

                    if re.fullmatch(self.translation_config.get_p_regex(), match_value):
                        self.logger.debug(
                            f"Skipping NER match as it is a protected placeholder: {match_value}"
                        )
                        return match_value

                    if re.fullmatch(
                        self.translation_config.get_ner_regex(), match_value
                    ):
                        self.logger.debug(
                            f"Skipping NER match as it is a newline placeholder: {match_value}"
                        )
                        return match_value

                    key = self.translation_config.get_ner_template(
                        len(ner_placeholders)
                    )
                    ner_placeholders[key] = match_value

                    return key

                context_text = ner_pattern.sub(_replace_ner_match, context_text)
            except re.error as e:
                self.logger.error(f"Error compiling NER pattern: {e}")
                raise

        # Step 3: Protect newlines
        context_text = context_text.replace(
            "\\n", self.translation_config.get_nl_template(0)
        )
        full_text = full_text.replace("\\n", self.translation_config.get_nl_template(0))

        return context_text, full_text, protected_placeholders, ner_placeholders

    def _restore_patterns(
        self,
        translated_text: TranslatedIniValueType,
        protected_placeholders: PlaceholdersType,
        ner_placeholders: PlaceholdersType,
    ) -> TranslatedIniValueType:
        """
        Restores protected and NER placeholders. NER can be restored as originals or translated.

        Args:
            translated_text (str): Translated text with placeholders.
            protected_placeholders (Dict[str, str]): Protected pattern placeholders.
            ner_placeholders (Dict[str, str]): NER placeholders.

        Returns:
            str: Text with placeholders restored.
        """
        result = translated_text

        # Step 1: Restore NER placeholders (to handle potential overlaps first)
        for key, original in ner_placeholders.items():
            result = re.sub(re.escape(key), original, result)

        # Step 2: Restore protected placeholders
        for key, value in protected_placeholders.items():
            result = re.sub(re.escape(key), value, result)

        # Step 3: Restore newlines
        result = result.replace(self.translation_config.get_nl_template(0), "\\n")

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

        nl_placeholder = self.translation_config.get_nl_template(0)
        paragraph_separator = f"{nl_placeholder}{nl_placeholder}"

        sentences: List[str] = []
        paragraphs = text.split(paragraph_separator)

        for paragraph in paragraphs:
            if paragraph.strip():
                sentences.extend(sent_tokenize(paragraph.strip(nl_placeholder)))

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
        ner_patterns: JSONNERListType,
    ) -> Tuple[TranslatedIniValueType, TranslatedIniValueType]:
        """
        Translates text, returning both context (NER protected) and full (NER translated) versions.

        Args:
            text (INIFIleValueType): Input text.
            translator (TranslatorCallableType): Translator function.
            ner_patterns (JSONNERListType): Patterns for NER protection.

        Returns:
            Tuple[str, str]: (context_version, full_version)
        """
        if not text:
            return text, text

        context_text, full_text, protected_placeholders, ner_placeholders = (
            self._protect_patterns(text, ner_patterns)
        )

        batch_texts = [context_text, full_text]  # Order: context first, full second
        batch_translated = translator(batch_texts)

        if len(batch_translated) != 2:
            self.logger.error(f"Expected 2 translations, got {len(batch_translated)}")
            raise ValueError(f"Expected 2 translations")

        context_translated_text, full_translated_text = batch_translated

        context_version = self._restore_patterns(
            context_translated_text, protected_placeholders, ner_placeholders
        )
        full_version = self._restore_patterns(
            full_translated_text, protected_placeholders, {}
        )

        return context_version, full_version
