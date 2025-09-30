import re
from dataclasses import dataclass, field

from src.type_defs import (
    ExcludeKeysType,
    ProtectedPatternsType,
)


@dataclass
class TranslationConfig:
    buffer_size: int = 50  # Number of lines for writing translations to file

    _length_language_ratio = {
        (
            "en",
            "ru",
        ): 1.5,  # English → Russian: text is usually longer. Or you can get it from ~ len(translated_tokens) / len(source_tokens)
        # Add other pairs as needed and select the coefficient based on the translation quality obtained
    }
    token_reserve: int = 20  # Reserve for BOS/EOS/special tokens

    exclude_keys: ExcludeKeysType = field(
        default_factory=lambda: (
            "HUD_Visor_DataDownload_DataCloseup_01,P",
            "HUD_Visor_DataDownload_DataCloseup_02,P",
            "HUD_Visor_DataDownload_DataCloseup_03,P",
            "HUD_Visor_DataDownload_DataCloseup_04,P",
            "F_Ind_HackingFluff,P",
            "DataHeist_IP_Generic",
            "ui_game_popup_error",
            "Text_Clovis_Safe_Contents_01",
            "test_special_chars",
        )
    )

    protected_patterns: ProtectedPatternsType = field(
        default_factory=lambda: (
            r"(%[a-zA-Z0-9\.\-_]+)",  # Variables like %X, %X1, %X23, etc.
            r"(#~[a-zA-Z0-9]+\([^\)]*\))",  # #~action(...), #~mission(...) and others
            r"(~[a-zA-Z0-9]+\([^\)]*\))",  # All actions like ~action(...), ~mission(...) and others
            r"(~[a-zA-Z0-9]+)",  # Simply ~mission, ~action and any other words starting with ~
            r"(aUEC|\u03BCSCU|μSCU|SCU)",  # Currency and units: aUEC, μSCU, SCU
            r"\[\s*[^]]*?\s*\]",  # Any words in square brackets, including brackets themselves
            r"(<\s*/?[^>]+>)",  # All tags <...> (opening and closing)
            r"(</\s*[^>]+>)",  # Closing tag </...>
            r"(<\s*[^/][^>]*\s*>)",  # All words in angle brackets <...>
        )
    )

    @classmethod
    def _get_template(cls, index: int, prefix: str) -> str:
        return "[%d%s]" % (index, prefix)

    @classmethod
    def get_p_template(cls, index: int) -> str:
        return cls._get_template(
            index, "LOCATION_OR_ACTIONS"
        )  # For an original model without fine tuning, the best template would be pp@pp or similar.

    @classmethod
    def get_ner_template(cls, index: int) -> str:
        return cls._get_template(index, "nn@nn")

    @classmethod
    def get_nl_template(cls) -> str:
        return "[0]"

    @classmethod
    def get_p_regex(cls) -> str:
        template = cls.get_p_template(0)
        return re.escape(template).replace("0", r"\d+")

    @classmethod
    def get_ner_regex(cls) -> str:
        template = cls.get_ner_template(0)
        return re.escape(template).replace("0", r"\d+")

    @classmethod
    def get_language_ratio(cls, src_lang: str, tgt_lang: str):
        return cls._length_language_ratio.get((src_lang, tgt_lang), 1.0)
