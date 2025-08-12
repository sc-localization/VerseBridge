from dataclasses import dataclass, field

from src.type_defs import (
    ExcludeKeysType,
    ProtectedPatternsType,
)


@dataclass
class TranslationConfig:
    buffer_size: int = 50

    min_tokens: int = 16
    length_scale_factors = {
        (
            "en",
            "ru",
        ): 1.5,  # English → Russian: text is usually longer. Or you can get it from ~ len(translated_tokens) / len(source_tokens)
        # Add other pairs as needed and select the coefficient based on the translation quality obtained
    }
    token_reserve: int = 10

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
    def get_scale_factor(cls, src_lang: str, tgt_lang: str):
        return cls.length_scale_factors.get((src_lang, tgt_lang), 1.0)
