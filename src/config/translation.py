from dataclasses import dataclass, field
from pathlib import Path

from src.type_defs import (
    ExcludeKeysType,
    ProtectedPatternsType,
    # TranslationPriorityType,
)
from .language import LanguageConfig
from .paths import PathConfig


@dataclass
class TranslationConfig:
    path_config: PathConfig
    lang_config: LanguageConfig

    translation_src_dir: Path = field(init=False)
    translation_dest_dir: Path = field(init=False)
    input_ini_file_path: Path = field(init=False)
    buffer_size: int = 50

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

    def __post_init__(self):
        self.translation_src_dir = (
            self.path_config.translation_dir / self.lang_config.src_lang.value
        )

        self.translation_dest_dir = (
            self.path_config.translation_dir / self.lang_config.tgt_lang.value
        )

        self.original_ini_file_path = self.path_config.ini_files["original"]
