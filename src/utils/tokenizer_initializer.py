from typing import Optional, cast
from transformers import AutoTokenizer

from src.config import ConfigManager
from src.type_defs import LoggerType, AppTaskType, InitializedTokenizerType


class TokenizerInitializer:
    def __init__(
        self,
        config: ConfigManager,
        logger: LoggerType,
        task: AppTaskType = "translation",
    ) -> None:
        """
        Initializes a new TokenizerInitializer instance.

        Args:
            config: The configuration.
            logger: The logger.
            mode: The mode ("translation" or "ner"). Defaults to "translation".
        """
        self.config = config
        self.logger = logger
        self.task = task

        self.tokenizer: Optional[InitializedTokenizerType] = None

    def initialize(self) -> InitializedTokenizerType:
        """
        Initializes and returns a tokenizer.
        Always uses the same tokenizer with the base model.

        Returns:
            InitlizedTokenizerType: A tokenizer instance.
        """

        model_name: str = (
            self.config.ner_config.model_name
            if self.task == "ner"
            else self.config.model_config.model_name
        )

        tokenizerParams = {
            "pretrained_model_name_or_path": model_name,
        }

        if self.task == "translation":
            tokenizerParams["src_lang"] = self.config.lang_config.src_nllb_lang_code

        try:
            self.tokenizer = cast(
                InitializedTokenizerType,
                AutoTokenizer.from_pretrained(**tokenizerParams),
            )

            self.logger.debug(
                f"Tokenizer initialized for model: {model_name}, mode: {self.task}"
            )

            return self.tokenizer
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {str(e)}")
            raise
