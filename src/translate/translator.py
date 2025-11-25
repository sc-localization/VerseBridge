import torch
from transformers import BatchEncoding
from typing import List, Optional, Any, Tuple, TypedDict
from tqdm import tqdm

from src.config import ConfigManager
from src.type_defs import (
    InitializedModelType,
    InitializedTokenizerType,
    LoggerType,
    INIFIleValueType,
    TranslatedIniValueType,
    CachedParamsType,
    TokenizerConfigType,
)
from src.utils import MemoryManager
from .text_processor import TextProcessor


class BatchWorkItemType(TypedDict):
    original_idx: int
    chunks: List[str]
    stride: int
    translated_chunks: List[str]


class Translator:
    def __init__(
        self,
        config: ConfigManager,
        model: InitializedModelType,
        tokenizer: InitializedTokenizerType,
        text_processor: TextProcessor,
        logger: LoggerType,
    ):
        self.config = config
        self.logger = logger
        self.tokenizer = tokenizer
        self.text_processor = text_processor
        self.model = model
        self.memory_manager = MemoryManager(self.logger)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._validate_initialization()

    def _validate_initialization(self) -> None:
        if not self.model:
            raise ValueError("Model must be initialized")
        if not self.tokenizer:
            raise ValueError("Tokenizer must be initialized")

    def _generate_translation(
        self,
        inputs: BatchEncoding,
        max_new_tokens: int,
        min_new_tokens: Optional[int] = None,
    ) -> Any:
        try:
            with torch.no_grad(), torch.amp.autocast("cuda"):
                return self.model.generate(
                    **inputs,  # type: ignore
                    generation_config=self.config.generation_config.to_generation_config(),
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                )
        except Exception as e:
            self.logger.error(f"Failed to generate translation: {str(e)}")
            raise

    def _calculate_token_limits(
        self, input_length: int, max_model_length: int, token_reserve: int
    ) -> tuple[Optional[int], int]:
        min_new_tokens = None
        max_new_tokens = max_model_length - input_length - token_reserve

        if max_new_tokens <= 0:
            self.logger.warning(
                f"Input length ({input_length}) exceeds model limit ({max_model_length})"
            )
            max_new_tokens = max(50, max_model_length - input_length)

        return min_new_tokens, max_new_tokens

    def _translate_single_text(
        self,
        texts: List[str],  # Changed to strictly List[str]
        tokenizer_args: TokenizerConfigType,
        cached_params: CachedParamsType,
    ) -> List[TranslatedIniValueType]:
        try:
            tgt_lang_token = self.config.lang_config.tgt_lang_token
            prefixed_texts = [f"{tgt_lang_token} {text}" for text in texts]

            tokens = self.tokenizer(prefixed_texts, **tokenizer_args).to(self.device)
            input_length = int(torch.sum(tokens.attention_mask, dim=1).max().item())  # type: ignore

            min_new_tokens, max_new_tokens = self._calculate_token_limits(
                input_length,
                cached_params["max_model_length"],
                cached_params["token_reserve"],
            )

            translated_tokens = self._generate_translation(
                tokens,
                max_new_tokens,
                min_new_tokens,
            )

            decoded_texts = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            return decoded_texts

        except Exception as e:
            self.logger.error(f"Failed to translate text: {str(e)}")
            raise

    def translate_batch(
        self, texts: List[INIFIleValueType], batch_size: int = 16
    ) -> List[TranslatedIniValueType]:
        """
        Translates a list of texts using efficient batching and SMART splitting.
        """
        if not texts:
            return []

        lang_config = self.config.lang_config
        translation_config = self.config.translation_config
        tokenizer_args = self.config.dataset_config.translation_dict

        max_model_length = tokenizer_args["max_length"]
        token_reserve = translation_config.token_reserve

        # Calculate safe length
        language_ratio = translation_config.get_language_ratio(
            lang_config.src_lang, lang_config.tgt_lang
        )
        # We don't need overlap subtraction anymore
        safe_input_len = int((max_model_length - token_reserve) / (1 + language_ratio))

        results: List[str] = [""] * len(texts)

        # 1. Prepare chunks using SMART split (Recursive)
        batch_work_items: List[BatchWorkItemType] = []

        self.logger.info(f"Preparing {len(texts)} texts for translation...")

        for idx, text in enumerate(texts):
            # Using the new SMART split method (no overlap return value)
            chunks = self.text_processor.split_text_smart(
                text, self.tokenizer, safe_input_len, tokenizer_args
            )

            batch_work_items.append(
                {
                    "original_idx": idx,
                    "chunks": chunks,
                    "stride": 0,  # Not used in smart split
                    "translated_chunks": [],
                }
            )

        # 2. Flatten chunks (Same logic as before)
        all_flat_chunks: List[Tuple[int, int, str]] = []  # (text_idx, chunk_idx, text)

        for item in batch_work_items:
            for c_idx, chunk in enumerate(item["chunks"]):
                all_flat_chunks.append((item["original_idx"], c_idx, chunk))

        self.logger.info(f"Total chunks to translate: {len(all_flat_chunks)}")

        # 3. Process in batches (Same logic as before)
        total_chunks = len(all_flat_chunks)
        cached_params: CachedParamsType = {
            "max_model_length": max_model_length,
            "token_reserve": token_reserve,
        }

        for i in tqdm(range(0, total_chunks, batch_size), desc="Translating batches"):
            current_batch_items = all_flat_chunks[i : i + batch_size]
            current_texts = [item[2] for item in current_batch_items]

            try:
                translated_batch = self._translate_single_text(
                    current_texts, tokenizer_args, cached_params
                )

                for j, translated_text in enumerate(translated_batch):
                    orig_idx = current_batch_items[j][0]
                    batch_work_items[orig_idx]["translated_chunks"].append(
                        translated_text
                    )

            except Exception as e:
                self.logger.error(f"Batch translation failed at index {i}: {e}")
                for j in range(len(current_texts)):
                    orig_idx = current_batch_items[j][0]
                    batch_work_items[orig_idx]["translated_chunks"].append("[ERROR]")

        # 4. Merge results using simple JOIN (Smart split preserves separators)
        for item in batch_work_items:
            # Simple join because split_text_smart kept the spaces/newlines
            final_text = "".join(item["translated_chunks"])
            results[item["original_idx"]] = final_text

        return results
