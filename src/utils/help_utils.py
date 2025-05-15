from pathlib import Path
from src.type_defs import (
    ArgLoggerType,
    HelpStringsDictType,
    LangCode,
    is_json_help_strings_dict_type,
)
from .file_utils import FileUtils


class HelpUtils:
    @staticmethod
    def get_help_strings(
        file_name: str, language: LangCode = LangCode.EN, logger: ArgLoggerType = None
    ) -> HelpStringsDictType:
        help_file = Path(__file__).parent.parent.parent / "scripts" / file_name
        file_utils = FileUtils(logger=logger)
        help_strings = file_utils.load_json(help_file)

        if not is_json_help_strings_dict_type(help_strings):
            raise TypeError(
                f"Expected JSONHelpStringsDictType, got {type(help_strings)}"
            )

        return help_strings.get(language, help_strings["en"])
