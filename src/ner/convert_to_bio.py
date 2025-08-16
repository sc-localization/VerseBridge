from tqdm import tqdm

from src.config import ConfigManager
from src.type_defs import (
    JSONDataNERListType,
    JSONDataConvertedToBIOListType,
    LoggerType,
    InitializedTokenizerType,
    CharToTokenType,
)


class BIOConverter:
    def __init__(
        self,
        config: ConfigManager,
        tokenizer: InitializedTokenizerType,
        logger: LoggerType,
    ):
        self.config = config
        self.logger = logger

        self.tokenizer = tokenizer

    def convert_to_bio(
        self, data: JSONDataNERListType
    ) -> JSONDataConvertedToBIOListType:
        """Converts the dataset to BIO format."""
        bio_data: JSONDataConvertedToBIOListType = []

        for item in tqdm(data, desc="Converting to BIO format"):
            text = item["text"]

            entities = sorted(
                item.get("entities", []), key=lambda x: (x["start"], x["end"])
            )

            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            labels = ["O"] * len(tokens)

            char_to_token: CharToTokenType = []
            char_idx = 0

            for i, _ in enumerate(tokens):
                token_text = self.tokenizer.decode(
                    [token_ids[i]], clean_up_tokenization_spaces=False  # type: ignore
                )

                token_len = len(token_text.replace("##", ""))
                char_to_token.append((char_idx, char_idx + token_len))
                char_idx += token_len

            overlaps = []

            for i, entity in enumerate(entities):
                for j in range(i + 1, len(entities)):
                    other = entities[j]

                    if not (
                        entity["end"] <= other["start"]
                        or other["end"] <= entity["start"]
                    ):
                        overlaps.append((i, j, entity, other))

            if overlaps:
                self.logger.warning(
                    f"Overlaps detected in example {item['id']}: {overlaps}"
                )

            for entity in entities:
                start_char, end_char, label = (
                    entity["start"],
                    entity["end"],
                    entity["label"],
                )

                if not (0 <= start_char < end_char <= len(text)):
                    self.logger.warning(
                        f"Invalid indices in entity {entity} for example {item['id']}"
                    )
                    continue

                for i, (token_start, token_end) in enumerate(char_to_token):
                    if token_end <= start_char or token_start >= end_char:
                        continue

                    if token_start >= start_char:
                        labels[i] = (
                            f"B-{label}" if token_start == start_char else f"I-{label}"
                        )

            bio_data.append({"id": item["id"], "tokens": tokens, "labels": labels})

        return bio_data
