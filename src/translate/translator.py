from typing import List
import torch
from transformers import BatchEncoding

from src.config import ConfigManager
from src.type_defs import (
    InitializedModelType,
    InitlizedTokenizerType,
    LoggerType,
    TranslatorCallableType,
    INIFIleValueType,
    TranslatedIniValueType,
)
from src.utils import MemoryManager
from .text_processor import TextProcessor


class Translator:
    def __init__(
        self,
        config: ConfigManager,
        model: InitializedModelType,
        tokenizer: InitlizedTokenizerType,
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
        target_lang_code: str,
        max_model_length: int = 512,
        min_tokens: int = 16,
        scale_factor: float = 1.8,
    ) -> str:
        if not self.model:
            self.logger.error("Model is not initialized")
            raise ValueError("Model must be initialized before generating translation")

        inputs_length = inputs.input_ids.shape[1]  # type: ignore
        max_new_tokens = int(min_tokens + scale_factor * inputs_length)  # type: ignore
        min_len = int(inputs_length * 1.1) if inputs_length > 10 else None  # type: ignore

        # Check for exceeding the model limit
        # Should not happen normally, as `max_inputs_allowed` should prevent such cases
        # If, despite `max_inputs_allowed`, the number of tokens exceeds `max_model_length`,
        # then decrease max_new_tokens
        total_length = max_new_tokens + inputs_length

        if total_length > max_model_length:
            self.logger.warning(
                f"Total length ({total_length}) exceeds model limit ({max_model_length}). Adjusting max_new_tokens."
            )
            max_new_tokens = (
                max_model_length
                - inputs_length
                - self.config.translation_config.token_reserve
            )

        try:
            with torch.no_grad(), torch.amp.autocast("cuda"):
                translated_tokens = self.model.generate(
                    **inputs,  # type: ignore
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_len,  # type: ignore
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(  # type: ignore
                        target_lang_code
                    ),
                    generation_config=self.config.generation_config.to_generation_config(),
                )

            generated_text = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            return generated_text[0]
        except Exception as e:
            self.logger.error(f"Failed to generate translation: {str(e)}")
            raise

    def create_translator(self) -> TranslatorCallableType:
        if not self.model or not self.tokenizer:
            raise ValueError(
                "Model and tokenizer must be initialized before creating translator"
            )

        def translate(text: INIFIleValueType) -> TranslatedIniValueType:
            self.tokenizer.src_lang = self.config.lang_config.src_nllb_lang_code
            self.tokenizer.tgt_lang = self.config.lang_config.tgt_nllb_lang_code

            src_lang = self.config.lang_config.src_lang
            tgt_lang = self.config.lang_config.tgt_lang

            tokenizer_args = self.config.dataset_config.to_dict()
            max_model_length = tokenizer_args["max_length"]

            tokens = self.tokenizer(text, **tokenizer_args)
            input_length = len(tokens["input_ids"][0])  # type: ignore

            min_tokens: int = self.config.translation_config.min_tokens
            scale_factor: float = self.config.translation_config.get_scale_factor(
                src_lang, tgt_lang
            )

            # The formula for `max_inputs_allowed` is calculated to limit the input text length, assuming that:
            #     inputs_length + max_new_tokens = inputs_length + (min_tokens + scale_factor * inputs_length) = min_tokens + (1 + scale_factor) * inputs_length <= max_model_length
            # From here:
            #     inputs_length <= (max_model_length - min_tokens) / (1 + scale_factor)
            max_inputs_allowed = int(
                (max_model_length - min_tokens) / (1 + scale_factor)
            )

            if input_length > max_inputs_allowed:
                self.logger.debug(
                    f"Text too long ({input_length} tokens), splitting..."
                )

                parts = self.text_processor.split_text(
                    text, self.tokenizer, max_inputs_allowed, tokenizer_args
                )

                translated_parts: List[str] = []

                for part in parts:
                    inputs = self.tokenizer(part, **tokenizer_args)
                    inputs = inputs.to(self.device)

                    translated_part = self.generate_translation(
                        inputs,
                        self.config.lang_config.tgt_nllb_lang_code,
                        max_model_length,
                        min_tokens,
                        scale_factor,
                    )

                    translated_parts.append(translated_part.strip())

                translated_text = " ".join(translated_parts)
            else:
                inputs = tokens.to(self.device)
                translated_text = self.generate_translation(
                    inputs,
                    self.config.lang_config.tgt_nllb_lang_code,
                    max_model_length,
                    min_tokens,
                    scale_factor,
                )

            return translated_text.strip()

        return translate
