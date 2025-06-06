import nltk
import re
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
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
        self.pattern = re.compile("|".join(protected_patterns))
        self.logger = logger

    def _protect_placeholders(
        self, text: INIFIleValueType
    ) -> Tuple[INIFIleValueType, PlaceholdersType]:
        """
        Protects special patterns in the text by replacing them with placeholders.

        Args:
            text (str): Input text.

        Returns:
            Tuple[str, Dict[str, str]]: A tuple containing the modified text and a dictionary of placeholders.
        """
        placeholders: PlaceholdersType = {}

        def replace_match(match: re.Match[str]) -> str:
            match_value = match.group(0)
            key = f"[{len(placeholders)}]"
            placeholders[key] = match_value

            return key

        modified_text = self.pattern.sub(replace_match, text)
        modified_text = modified_text.replace("\\n", "[NL]")

        return modified_text, placeholders

    def _restore_placeholders(
        self, translated_text: TranslatedIniValueType, placeholders: PlaceholdersType
    ) -> TranslatedIniValueType:
        """
        Restores protected patterns in the translated text by replacing placeholders with original values.

        Args:
            translated_text (str): The translated text.
            placeholders (Dict[str, str]): A dictionary of placeholders and their corresponding values.

        Returns:
            str: The text with placeholders replaced with original values.
        """
        result = translated_text

        for key, value in placeholders.items():
            result = re.sub(re.escape(key), value, result)

        return result.replace("[NL]", "\\n")

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
        self, text: INIFIleValueType, translator: TranslatorCallableType
    ) -> TranslatedIniValueType:
        if not text:
            return text

        modified_text, placeholders = self._protect_placeholders(text)
        translated_text = translator(modified_text)

        return self._restore_placeholders(translated_text, placeholders)
