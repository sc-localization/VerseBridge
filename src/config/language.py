from dataclasses import dataclass
from typing import ClassVar, List

from src.type_defs import (
    LangCode,
    MappedCode,
    LangMapType,
)


@dataclass(frozen=True)
class LanguageConfig:
    src_lang: LangCode = LangCode.EN
    tgt_lang: LangCode = LangCode.RU

    _lang_map: ClassVar[LangMapType] = {
        LangCode.EN: MappedCode.ENG_LATN,
        LangCode.RU: MappedCode.RUS_CYRL,
        # Add other languages as needed from src/type_defs/custom_types.py
        # https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-FLORES-200
    }

    @classmethod
    def _get_nllb_code(cls, lang: LangCode) -> MappedCode:
        """Returns the NLLB language code based on ISO 639-1."""
        if lang not in cls._lang_map:
            raise ValueError(
                f"Language '{lang}' is not supported. Please add it to _lang_map."
            )

        return cls._lang_map[lang]

    @property
    def src_nllb_lang_code(self) -> MappedCode:
        """Returns the NLLB language code for the source language."""
        return self._get_nllb_code(self.src_lang)

    @property
    def tgt_nllb_lang_code(self) -> MappedCode:
        """Returns the NLLB language code for the target language."""
        return self._get_nllb_code(self.tgt_lang)

    @property
    def available_languages(self) -> List[LangCode]:
        """Returns a list of available languages (ISO 639-1 codes)."""
        return list(self._lang_map.keys())
