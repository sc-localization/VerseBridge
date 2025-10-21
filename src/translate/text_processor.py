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
                            f"Skipping NER match as it is a existing placeholder: {match_value}"
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
        nl_key = self.translation_config.get_nl_template()
        context_text = context_text.replace("\\n", nl_key)
        full_text = full_text.replace("\\n", nl_key)

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
        result = result.replace(self.translation_config.get_nl_template(), "\\n")

        return result

    def _get_token_count(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_args: TokenizerConfigType,
    ) -> int:
        """Calculate the token count for a given text."""
        tokens = tokenizer(text, **tokenizer_args)

        return len(tokens["input_ids"][0])  # type: ignore

    def _truncate_text(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        tokenizer_args: TokenizerConfigType,
    ) -> str:
        """Truncate text to fit within max_tokens."""
        tokens = tokenizer(text, **tokenizer_args)
        truncated_ids = tokens["input_ids"][0][:max_tokens]  # type: ignore

        return tokenizer.decode(truncated_ids, skip_special_tokens=True)

    def _split_by_midpoint(self, text: str) -> List[str]:
        """Split text at a safe midpoint (preferably at a space) if it exceeds max_tokens."""
        mid_point = len(text) // 2
        safe_split_index = text.find(" ", mid_point - 50, mid_point + 50)

        if safe_split_index == -1:
            safe_split_index = mid_point

        return [text[:safe_split_index].strip(), text[safe_split_index:].strip()]

    def _split_recursively(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        tokenizer_args: TokenizerConfigType,
        depth: int = 0,
        max_depth: int = 5,
    ) -> List[str]:
        """
        Recursively split text into chunks smaller than max_tokens.
        Prioritizes splitting by newlines, then by midpoint if necessary.
        """
        if depth >= max_depth:
            self.logger.warning(
                f"Max split depth {max_depth} reached. Truncating text."
            )

            return [self._truncate_text(text, tokenizer, max_tokens, tokenizer_args)]

        # Check if text is already within token limit
        token_count = self._get_token_count(text, tokenizer, tokenizer_args)

        if token_count <= max_tokens:
            return [text]

        # Try splitting by newline placeholders
        nl_placeholder = re.escape(self.translation_config.get_nl_template())
        parts = re.split(f"{nl_placeholder}+", text)
        parts = [part.strip() for part in parts if part.strip()]

        if len(parts) > 1:
            chunks = []

            for part in parts:
                part_token_count = self._get_token_count(
                    part, tokenizer, tokenizer_args
                )

                if part_token_count > max_tokens:
                    chunks.extend(
                        self._split_recursively(
                            part,
                            tokenizer,
                            max_tokens,
                            tokenizer_args,
                            depth + 1,
                            max_depth,
                        )
                    )
                else:
                    chunks.append(part)

            return chunks

        # Fallback to midpoint splitting
        self.logger.debug("No newline split possible. Splitting by midpoint.")

        parts = self._split_by_midpoint(text)
        chunks = []

        for part in parts:
            part_token_count = self._get_token_count(part, tokenizer, tokenizer_args)

            if part_token_count > max_tokens:
                chunks.extend(
                    self._split_recursively(
                        part,
                        tokenizer,
                        max_tokens,
                        tokenizer_args,
                        depth + 1,
                        max_depth,
                    )
                )
            else:
                chunks.append(part)

        return chunks

    def split_text(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        tokenizer_args: TokenizerConfigType,
        max_depth: int = 5,
    ) -> List[str]:
        """
        Split text into chunks smaller than max_tokens, optimizing for performance.
        Sentences are grouped efficiently, and only oversized segments are split recursively.
        """
        # Early return for short texts
        if self._get_token_count(text, tokenizer, tokenizer_args) <= max_tokens:
            return [text]

        chunks = []
        current_chunk = []
        current_token_count = 0

        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentence_token_count = self._get_token_count(
                sentence, tokenizer, tokenizer_args
            )

            if sentence_token_count > max_tokens:
                # Flush current chunk if any
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                # Split oversized sentence recursively
                chunks.extend(
                    self._split_recursively(
                        sentence, tokenizer, max_tokens, tokenizer_args, 0, max_depth
                    )
                )
            elif current_token_count + sentence_token_count > max_tokens:
                # Flush current chunk and start new one
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_token_count = sentence_token_count
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_token_count += sentence_token_count

        # Append final chunk if any
        if current_chunk:
            chunks.append(" ".join(current_chunk))

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

        texts = [context_text, full_text]  # Order: context first, full second
        translated_texts = translator(texts)

        if len(translated_texts) != 2:
            self.logger.error(f"Expected 2 translations, got {len(translated_texts)}")
            raise ValueError(f"Expected 2 translations")

        context_translated_text, full_translated_text = translated_texts

        context_version = self._restore_patterns(
            context_translated_text, protected_placeholders, ner_placeholders
        )
        full_version = self._restore_patterns(
            full_translated_text, protected_placeholders, {}
        )

        return context_version, full_version
