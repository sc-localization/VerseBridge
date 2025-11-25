import nltk
import re
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizerBase

from src.config import ConfigManager
from src.type_defs import (
    PlaceholdersType,
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

    def protect_patterns(
        self, text: INIFIleValueType, ner_patterns: Optional[JSONNERListType] = None
    ) -> Tuple[INIFIleValueType, INIFIleValueType, PlaceholdersType, PlaceholdersType]:
        """
        Protects both protected patterns and NER entities with unique placeholders.
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
                        return match_value

                    if re.fullmatch(
                        self.translation_config.get_ner_regex(), match_value
                    ):
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

    def restore_patterns(
        self,
        translated_text: TranslatedIniValueType,
        protected_placeholders: PlaceholdersType,
        ner_placeholders: PlaceholdersType,
    ) -> TranslatedIniValueType:
        """
        Restores protected and NER placeholders.
        """
        result = translated_text

        # Step 1: Restore NER placeholders
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
        tokens = tokenizer(text, **tokenizer_args)
        return len(tokens["input_ids"][0])  # type: ignore

    def split_text_smart(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        tokenizer_args: TokenizerConfigType,
        depth: int = 0,
        max_depth: int = 10,
    ) -> List[str]:
        """
        Recursively splits text at safe boundaries (newlines, sentences, spaces)
        PRESERVING the separators in the chunks.
        Designed to be rejoined simply with "".join().
        """
        # 1. Check if text fits
        if self._get_token_count(text, tokenizer, tokenizer_args) <= max_tokens:
            return [text]

        if depth >= max_depth:
            self.logger.warning(f"Max split depth reached for text: {text[:50]}...")
            # Fallback: hard cut (not ideal but avoids infinite recursion)
            return [text]

        # 2. Strategy: Find the best split point near the middle
        mid = len(text) // 2
        # Define search window (middle 50% of text)
        start_search = len(text) // 4
        end_search = start_search * 3

        best_split_idx = -1

        # Priority 1: Newline placeholder [0]
        nl_template = self.translation_config.get_nl_template()
        # Find all occurrences of nl_template
        nl_indices = [m.start() for m in re.finditer(re.escape(nl_template), text)]

        # Find closest to mid
        closest_dist = float("inf")
        for idx in nl_indices:
            # We want to include the newline in the first chunk, so split AFTER it
            split_candidate = idx + len(nl_template)

            if start_search <= split_candidate <= end_search:
                dist = abs(split_candidate - mid)

                if dist < closest_dist:
                    closest_dist = dist
                    best_split_idx = split_candidate

        # Priority 2: Sentence boundary (. followed by space)
        if best_split_idx == -1:
            # Simple heuristic for sentence end.
            # Note: This might be less accurate than NLTK but easier to control indices.
            # We look for ". "
            matches = [m.start() for m in re.finditer(r"\.\s", text)]
            closest_dist = float("inf")

            for idx in matches:
                # Include ". " in the first chunk (or at least the dot)
                split_candidate = idx + 1  # Split after dot, keep space in next?
                # Better: Split after space -> idx + 2. "Sentence. " | "Next"
                split_candidate = idx + 2

                if start_search <= split_candidate <= end_search:
                    dist = abs(split_candidate - mid)

                    if dist < closest_dist:
                        closest_dist = dist
                        best_split_idx = split_candidate

        # Priority 3: Any Space
        if best_split_idx == -1:
            space_indices = [m.start() for m in re.finditer(r"\s", text)]
            closest_dist = float("inf")

            for idx in space_indices:
                # Split after space so space stays with first chunk
                split_candidate = idx + 1

                if start_search <= split_candidate <= end_search:
                    dist = abs(split_candidate - mid)

                    if dist < closest_dist:
                        closest_dist = dist
                        best_split_idx = split_candidate

        # Priority 4: Hard cut at middle (if no separators found)
        if best_split_idx == -1:
            best_split_idx = mid

        # 3. Perform Split
        left_part = text[:best_split_idx]
        right_part = text[best_split_idx:]

        # 4. Recursion
        # Process left part
        chunks = []
        chunks.extend(
            self.split_text_smart(
                left_part, tokenizer, max_tokens, tokenizer_args, depth + 1, max_depth
            )
        )
        # Process right part
        chunks.extend(
            self.split_text_smart(
                right_part, tokenizer, max_tokens, tokenizer_args, depth + 1, max_depth
            )
        )

        return chunks
