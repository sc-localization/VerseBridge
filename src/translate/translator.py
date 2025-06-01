import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding

from src.config import ConfigManager
from src.type_defs import (
    InitializedModelType,
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
        tokenizer: PreTrainedTokenizerBase,
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
        min_tokens: int = 32,
        scale_factor: float = 3.0,
    ) -> str:
        if not self.model:
            self.logger.error("Model is not initialized")
            raise ValueError("Model must be initialized before generating translation")

        inputs_length = inputs.input_ids.shape[1]  # type: ignore
        max_new_tokens = int(min_tokens + scale_factor * inputs_length)  # type: ignore
        min_len = int(inputs_length * 1.1) if inputs_length > 10 else None  # type: ignore

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

            tokenizer_args = self.config.dataset_config.to_dict()
            max_model_length = tokenizer_args["max_length"]

            tokens = self.tokenizer(text, **tokenizer_args)
            input_length = len(tokens["input_ids"][0])  # type: ignore

            if input_length > max_model_length:
                self.logger.info(f"Text too long ({input_length} tokens), splitting...")

                parts = self.text_processor.split_text(
                    text, self.tokenizer, max_model_length
                )

                inputs = self.tokenizer(parts, **tokenizer_args)
                inputs = inputs.to(self.device)

                translated_text = self.generate_translation(
                    inputs, self.config.lang_config.tgt_nllb_lang_code
                )
            else:
                inputs = tokens.to(self.device)
                translated_text = self.generate_translation(
                    inputs, self.config.lang_config.tgt_nllb_lang_code
                )

            return translated_text

        return translate
