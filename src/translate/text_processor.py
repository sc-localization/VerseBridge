import re
from typing import List, Tuple
from transformers import PreTrainedTokenizerBase

from src.type_defs import (
    ProtectedPatternsType,
    PlaceholdersType,
    TranslatorCallableType,
)


class TextProcessor:
    def __init__(self, protected_patterns: ProtectedPatternsType):
        self.pattern = re.compile("|".join(protected_patterns))

    def protect_placeholders(
        self, text: str
    ) -> Tuple[str, PlaceholdersType]:
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

    def restore_placeholders(
        self, translated_text: str, placeholders: PlaceholdersType
    ) -> str:
        """
        Restores protected patterns in the translated text by replacing placeholders with original values.

        Args:
            translated_text (str): The translated text.
            placeholders (Dict[str, str]): A dictionary of placeholders and their corresponding values.

        Returns:
            str: The text with placeholders replaced with original values.
        """
        result: str = translated_text

        for key, value in placeholders.items():
            result = re.sub(re.escape(key), value, result)

        return result.replace("[NL]", "\\n")

    def split_text(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
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
        sentences = text.split(". ")
        chunks: List[str] = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(tokenizer(sentence)["input_ids"])  # type: ignore

            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence + ". "
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def translate_text(self, text: str, translator: TranslatorCallableType) -> str:
        if not text:
            return text

        modified_text, placeholders = self.protect_placeholders(text)
        translated_text = translator(modified_text)

        return self.restore_placeholders(translated_text, placeholders)
