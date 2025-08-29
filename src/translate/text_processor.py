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

    def _split_recursively(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        tokenizer_args: TokenizerConfigType,
        depth: int,
        max_depth: int,
    ) -> List[str]:
        """
        Recursive helper to split a text that is known to be too long.
        This version splits by newline placeholders first, as it's a less aggressive
        and more semantically meaningful splitting strategy than using spaces.
        """
        if depth >= max_depth:
            self.logger.warning(
                f"Max split depth {max_depth} reached. Truncating oversized text segment."
            )
            tokens = tokenizer(text, **tokenizer_args)
            truncated_ids = tokens["input_ids"][0][:max_tokens]  # type: ignore

            return [tokenizer.decode(truncated_ids, skip_special_tokens=True)]

        nl_placeholder = self.translation_config.get_nl_template()
        nl_placeholder_escaped = re.escape(nl_placeholder)

        split_patterns = [
            f"({nl_placeholder_escaped}{nl_placeholder_escaped})",  # Double newline (\n\n -> [0][0])
            f"({nl_placeholder_escaped})",  # Single newline (\n -> [0])
            r"([,;:])",
        ]

        for pattern in split_patterns:
            if not pattern:
                continue

            try:
                parts: List[str] = re.split(pattern, text)
            except re.error as e:
                self.logger.error(f"Regex error on pattern '{pattern}': {e}")
                continue

            if len(parts) > 1:
                merged_parts: List[str] = []

                for i in range(0, len(parts), 2):
                    part = parts[i]

                    if i + 1 < len(parts):
                        part += parts[i + 1]

                    if part:
                        merged_parts.append(part)

                # Recursively process the new, smaller parts
                sub_chunks: List[str] = []

                for part in merged_parts:
                    part_token_len = len(
                        tokenizer(part, **tokenizer_args)["input_ids"][0]  # type: ignore
                    )

                    if part_token_len > max_tokens:
                        sub_chunks.extend(
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
                        sub_chunks.append(part)

                return sub_chunks

        self.logger.warning(
            "Could not split text further with available patterns. Truncating."
        )

        tokens = tokenizer(text, **tokenizer_args)
        truncated_ids = tokens["input_ids"][0][:max_tokens]  # type: ignore

        return [tokenizer.decode(truncated_ids, skip_special_tokens=True)]

    def split_text(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        tokenizer_args: TokenizerConfigType,
        max_depth: int = 5,
    ) -> List[str]:
        """
        Splits text into chunks that are smaller than max_tokens.

        This refactored version improves performance by:
        1.  Tokenizing each sentence only once.
        2.  Avoiding repeated string concatenation and re-tokenization in a loop.
        3.  Grouping sentences efficiently before performing more complex recursive splits
            only on sentences that are individually too long.
        """
        final_chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_chunk_token_count = 0

        sentences = sent_tokenize(text)

        for sentence in sentences:
            sentence_token_len = len(
                tokenizer(sentence, **tokenizer_args)["input_ids"][0]  # type: ignore
            )

            # Case 1: A single sentence is too long and must be split recursively.
            if sentence_token_len > max_tokens:
                if current_chunk_sentences:
                    final_chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_chunk_token_count = 0

                sub_chunks = self._split_recursively(
                    sentence, tokenizer, max_tokens, tokenizer_args, 0, max_depth
                )
                final_chunks.extend(sub_chunks)
                continue

            # Case 2: Adding the next sentence would make the current chunk too long.
            if current_chunk_token_count + sentence_token_len > max_tokens:
                if current_chunk_sentences:
                    final_chunks.append(" ".join(current_chunk_sentences))

                current_chunk_sentences = [sentence]
                current_chunk_token_count = sentence_token_len
            # Case 3: The sentence fits, so add it to the current chunk.
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_token_count += sentence_token_len

        if current_chunk_sentences:
            final_chunks.append(" ".join(current_chunk_sentences))

        return final_chunks

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
