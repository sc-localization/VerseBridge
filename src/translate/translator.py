import torch
from transformers import BatchEncoding
from typing import List, Optional, Any

from src.config import ConfigManager
from src.type_defs import (
    InitializedModelType,
    InitializedTokenizerType,
    LoggerType,
    TranslatorCallableType,
    INIFIleValueType,
    TranslatedIniValueType,
    CachedParamsType,
    TokenizerConfigType,
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

        self._validate_initialization()

    def _validate_initialization(self) -> None:
        """Validate that required components are properly initialized."""
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
        """
        Generate translation for given inputs.
        """
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
        """
        Calculate min and max new tokens for generation.
        """
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
        text: INIFIleValueType,
        tokenizer_args: TokenizerConfigType,
        cached_params: CachedParamsType,
    ) -> TranslatedIniValueType:
        """
        Translate a single text.
        """
        try:
            tgt_lang_token = self.config.lang_config.tgt_lang_token
            text = f"{tgt_lang_token} {text}"  # <2tgt_lang> text

            tokens = self.tokenizer(text, **tokenizer_args).to(self.device)
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

            translated_text = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0].strip()

            return translated_text

        except Exception as e:
            self.logger.error(f"Failed to translate text: '{text[:100]}...' - {str(e)}")
            raise

    def _should_split_text(
        self, text: str, tokenizer_args: TokenizerConfigType, max_inputs_allowed: int
    ) -> bool:
        """
        Check if text needs to be split due to length constraints.
        """
        tokens = self.tokenizer(text, **tokenizer_args)
        input_length = len(tokens["input_ids"][0])  # type: ignore

        return input_length >= max_inputs_allowed

    def create_translator(self) -> TranslatorCallableType:
        """
        Create a translator callable function.
        """
        lang_config = self.config.lang_config
        translation_config = self.config.translation_config
        dataset_config = self.config.dataset_config

        src_lang = lang_config.src_lang
        tgt_lang = lang_config.tgt_lang

        token_reserve = translation_config.token_reserve
        tokenizer_args = dataset_config.to_dict()
        max_model_length = tokenizer_args["max_length"]

        language_ratio = translation_config.get_language_ratio(src_lang, tgt_lang)

        cached_params: CachedParamsType = {
            "max_model_length": max_model_length,
            "token_reserve": token_reserve,
        }

        max_inputs_allowed = int(
            (max_model_length - token_reserve) / (1 + language_ratio)
        )

        def translate(texts: List[INIFIleValueType]) -> List[TranslatedIniValueType]:
            if not texts:
                self.logger.warning("Empty input text list provided")

                return []

            translated_texts: List[str] = []

            try:
                for text in texts:
                    if self._should_split_text(
                        text, tokenizer_args, max_inputs_allowed
                    ):
                        self.logger.debug("Text too long, splitting...")

                        chunks = self.text_processor.split_text(
                            text, self.tokenizer, max_inputs_allowed, tokenizer_args
                        )

                        translated_chunks = [
                            self._translate_single_text(
                                chunk, tokenizer_args, cached_params
                            )
                            for chunk in chunks
                        ]

                        translated_texts.append(" ".join(translated_chunks))
                    else:
                        translated_text = self._translate_single_text(
                            text, tokenizer_args, cached_params
                        )

                        translated_texts.append(translated_text)

                return translated_texts

            except Exception as e:
                self.logger.error(f"Translation failed: {str(e)}")
                raise

        return translate
