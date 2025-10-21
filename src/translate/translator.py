import torch
from transformers import BatchEncoding
from typing import List, Optional, Any
from tqdm import tqdm

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
        texts: List[INIFIleValueType],
        tokenizer_args: TokenizerConfigType,
        cached_params: CachedParamsType,
    ) -> List[TranslatedIniValueType]:
        """
        Translate a single text or batch of texts.
        """
        try:
            tgt_lang_token = self.config.lang_config.tgt_lang_token
            # Добавляем языковой токен к каждому тексту
            prefixed_texts = [f"{tgt_lang_token} {text}" for text in texts]

            # Токенизация батча
            tokens = self.tokenizer(prefixed_texts, **tokenizer_args).to(self.device)
            input_length = int(torch.sum(tokens.attention_mask, dim=1).max().item())  # type: ignore

            min_new_tokens, max_new_tokens = self._calculate_token_limits(
                input_length,
                cached_params["max_model_length"],
                cached_params["token_reserve"],
            )

            # Генерация переводов
            translated_tokens = self._generate_translation(
                tokens,
                max_new_tokens,
                min_new_tokens,
            )

            # Декодирование результатов
            decoded_texts = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            return decoded_texts

        except Exception as e:
            self.logger.error(f"Failed to translate text: {str(e)}")
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
        tokenizer_args = dataset_config.translation_dict
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
            """
            Translate a list of texts, handling splitting when necessary.
            """
            if not texts:
                self.logger.warning("Empty input text list provided")
                return []

            translated_texts: List[TranslatedIniValueType] = []

            # Since almost all texts will have roughly the same length, we assume that at least one of them needs to be split into parts.
            is_any_text_too_long = any(
                self._should_split_text(text, tokenizer_args, max_inputs_allowed)
                for text in texts
            )

            try:
                if is_any_text_too_long:
                    self.logger.debug(
                        "Some texts are too long, processing with splitting..."
                    )

                    for _, text in enumerate(texts):
                        chunks = self.text_processor.split_text(
                            text, self.tokenizer, max_inputs_allowed, tokenizer_args
                        )

                        self.logger.debug(
                            f"Text: {text} \nsplit into {len(chunks)} chunks"
                        )

                        translated_chunks: List[TranslatedIniValueType] = []

                        # Translate chunks one by one, as translating a batch leads to dependence on the order of the chunks.
                        for chunk in tqdm(chunks, desc=f"Translating chunks"):
                            chunk_result = self._translate_single_text(
                                [chunk], tokenizer_args, cached_params
                            )[0]

                            translated_chunks.append(chunk_result)

                        translated_text = "".join(translated_chunks)
                        translated_texts.append(translated_text)

                else:
                    translated_texts = self._translate_single_text(
                        texts, tokenizer_args, cached_params
                    )

                return translated_texts

            except Exception as e:
                self.logger.error(f"Translation failed: {str(e)}")
                raise

        return translate
