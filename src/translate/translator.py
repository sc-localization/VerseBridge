import torch
from transformers import BatchEncoding
from typing import List, Any, Optional

from src.config import ConfigManager
from src.type_defs import (
    InitializedModelType,
    InitializedTokenizerType,
    LoggerType,
    TranslatorCallableType,
    INIFIleValueType,
    TranslatedIniValueType,
    GeneratedKwargsType,
)
from src.utils import MemoryManager
from .text_processor import TextProcessor


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

    def generate_translation(
        self,
        inputs: BatchEncoding,
        generated_kwargs: GeneratedKwargsType,
    ) -> Any:
        if not self.model:
            self.logger.error("Model is not initialized")
            raise ValueError("Model must be initialized before generating translation")

        input_lengths = torch.sum(inputs.attention_mask, dim=1)  # type: ignore
        max_input_length = torch.max(input_lengths).item()

        max_model_length = generated_kwargs["max_model_length"]
        min_tokens = generated_kwargs["min_tokens"]
        scale_factor = generated_kwargs["scale_factor"]
        target_lang_code = generated_kwargs["tgt_nllb_lang_code"]

        max_new_tokens = max(
            [int(min_tokens + scale_factor * l.item()) for l in input_lengths]
        )

        # Check for exceeding the model limit
        # Should not happen normally, as `max_inputs_allowed` should prevent such cases
        # If, despite `max_inputs_allowed`, the number of tokens exceeds `max_model_length`,
        # then decrease max_new_tokens
        total_lengths = [l + max_new_tokens for l in input_lengths]

        if any(t > max_model_length for t in total_lengths):
            self.logger.warning(
                f"Total length ({total_lengths}) exceeds model limit ({max_model_length}). Adjusting max_new_tokens."
            )
            max_new_tokens = (
                max_model_length
                - max_input_length
                - self.config.translation_config.token_reserve
            )

        min_new_tokens = max(
            1, int(max_input_length * 0.6)
        )  # At least 60% of the input length

        try:
            with torch.no_grad(), torch.amp.autocast("cuda"):
                translated_tokens = self.model.generate(
                    **inputs,  # type: ignore
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,  # type: ignore
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(  # type: ignore
                        target_lang_code
                    ),
                    generation_config=self.config.generation_config.to_generation_config(),
                )

            return translated_tokens
        except Exception as e:
            self.logger.error(f"Failed to generate translation: {str(e)}")
            raise

    def create_translator(self) -> TranslatorCallableType:
        if not self.model or not self.tokenizer:
            raise ValueError(
                "Model and tokenizer must be initialized before creating translator"
            )

        src_nllb_lang_code = self.config.lang_config.src_nllb_lang_code
        tgt_nllb_lang_code = self.config.lang_config.tgt_nllb_lang_code
        src_lang = self.config.lang_config.src_lang
        tgt_lang = self.config.lang_config.tgt_lang

        tokenizer_args = self.config.dataset_config.to_dict()
        max_model_length = tokenizer_args["max_length"]

        min_tokens: int = self.config.translation_config.min_tokens
        scale_factor: float = self.config.translation_config.get_scale_factor(
            src_lang, tgt_lang
        )

        # The formula for `max_inputs_allowed` is calculated to limit the input text length, assuming that:
        #     inputs_length + max_new_tokens = inputs_length + (min_tokens + scale_factor * inputs_length) = min_tokens + (1 + scale_factor) * inputs_length <= max_model_length
        # From here:
        #     inputs_length <= (max_model_length - min_tokens) / (1 + scale_factor)

        max_inputs_allowed = int((max_model_length - min_tokens) / (1 + scale_factor))

        generate_kwargs: GeneratedKwargsType = {
            "max_model_length": max_model_length,
            "min_tokens": min_tokens,
            "scale_factor": scale_factor,
            "tgt_nllb_lang_code": tgt_nllb_lang_code,
        }

        def translate(texts: List[INIFIleValueType]) -> List[TranslatedIniValueType]:
            if not texts:
                self.logger.warning(
                    "Empty input text list provided, returning empty list"
                )
                return []

            self.tokenizer.src_lang = src_nllb_lang_code
            self.tokenizer.tgt_lang = tgt_nllb_lang_code

            try:
                batch_tokens = self.tokenizer(texts, **tokenizer_args).to(self.device)
                input_lengths = [
                    len(ids) for ids in batch_tokens["input_ids"]  # type:ignore
                ]
            except Exception as e:
                self.logger.error(f"Tokenization failed: {str(e)}")
                raise

            translated_texts: List[Optional[str]] = [None] * len(texts)
            batch_texts: List[str] = []
            batch_indices: List[int] = []

            for i, text in enumerate(texts):
                input_length = input_lengths[i]

                if input_length > max_inputs_allowed:
                    self.logger.debug(
                        f"Text {i} too long ({input_length} tokens), splitting..."
                    )

                    parts = self.text_processor.split_text(
                        text, self.tokenizer, max_inputs_allowed, tokenizer_args
                    )

                    part_translations: List[str] = []

                    for part in parts:
                        part_tokens = self.tokenizer(part, **tokenizer_args).to(
                            self.device
                        )

                        translated_part = self.generate_translation(
                            part_tokens, generate_kwargs
                        )

                        translated_text = self.tokenizer.batch_decode(
                            translated_part, skip_special_tokens=True
                        )[0]

                        part_translations.append(translated_text.strip())

                    translated_texts.append(" ".join(part_translations))
                else:
                    batch_texts.append(text)
                    batch_indices.append(i)

            if batch_texts:
                batch_tokens = self.tokenizer(batch_texts, **tokenizer_args).to(
                    self.device
                )

                batch_outputs = self.generate_translation(batch_tokens, generate_kwargs)

                decoded_batch = self.tokenizer.batch_decode(
                    batch_outputs, skip_special_tokens=True
                )

                for idx, result in zip(batch_indices, decoded_batch):
                    translated_texts[idx] = result.strip()

            return [text for text in translated_texts if text is not None]

        return translate
