from typing import Optional, cast
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import ConfigManager
from src.type_defs import LoggerType, MappedCode


class TokenizerInitializer:
    def __init__(self, config: ConfigManager, logger: LoggerType):
        self.config = config
        self.logger = logger
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    def initialize(self) -> PreTrainedTokenizerBase:
        """
        Initializes and returns a tokenizer.
        Always uses the same tokenizer with the base model.

        Returns:
        PreTrainedTokenizerBase: A tokenizer instance.
        """

        model_name: str = self.config.model_config.model_name
        src_nllb_lang_code: MappedCode = self.config.lang_config.src_nllb_lang_code

        try:
            self.tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(  # type: ignore
                    pretrained_model_name_or_path=model_name,
                    src_lang=src_nllb_lang_code,
                ),
            )

            self.logger.debug(
                f"Tokenizer initialized for model: {self.config.model_config.model_name}"
            )

            return self.tokenizer
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {str(e)}")
            raise
