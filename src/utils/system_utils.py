import locale

from src.type_defs import LangCode


class SystemUtils:
    @staticmethod
    def get_system_lang_code() -> LangCode:
        """
        Determines the system's language code.

        Returns:
            str: Language code (e.g., 'en', 'ru') or 'en' if undetermined.
        """
        try:
            lang_code: str | None = locale.getdefaultlocale()[0]

            if not lang_code:
                return LangCode.EN

            result: str = lang_code.split(".")[0].split("_")[0]

            return LangCode(result) if result in LangCode else LangCode.EN
        except Exception:
            return LangCode.EN
