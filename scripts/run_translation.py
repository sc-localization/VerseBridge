import argparse
from pathlib import Path
import torch

from src.config import ConfigManager
from src.utils import AppLogger, CustomHelpFormatter, SystemUtils, HelpUtils
from src.translate import TranslationPipeline
from src.type_defs import (
    LangCode,
    LoggerType,
)


def parse_args() -> argparse.Namespace:
    logger = AppLogger("cli_parser").get_logger
    system_utils = SystemUtils()
    help_utils = HelpUtils()

    system_language = system_utils.get_system_lang_code()
    help_strings = help_utils.get_help_strings(
        "help_strings_translate.json", system_language, logger
    )

    parser = argparse.ArgumentParser(
        description=help_strings["description"],
        formatter_class=CustomHelpFormatter,
        epilog=help_strings["epilog"],
    )

    # Language Parameters
    lang_group = parser.add_argument_group(help_strings["lang_group_title"])
    lang_group.add_argument(
        "--src-lang",
        default=LangCode.EN,
        type=LangCode,
        choices=LangCode,
        metavar="",
        help=help_strings["src_lang_help"],
    )
    lang_group.add_argument(
        "--tgt-lang",
        default=LangCode.RU,
        type=LangCode,
        choices=LangCode,
        metavar="",
        help=help_strings["tgt_lang_help"],
    )

    # Model Parameters
    model_group = parser.add_argument_group(help_strings["model_group_title"])
    model_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        metavar="",
        help=help_strings["model_path_help"],
    )

    # Path Parameters
    path_group = parser.add_argument_group(help_strings["path_group_title"])

    path_group.add_argument(
        "--translated-file-name",
        type=str,
        default=None,
        metavar="",
        help=help_strings["translated_file_name_help"],
    )
    path_group.add_argument(
        "--input-file",
        type=str,
        default=None,
        metavar="",
        help=help_strings["input_file_help"],
    )

    args = parser.parse_args()

    # Validate language codes
    valid_langs = ConfigManager().lang_config.available_languages

    if args.src_lang not in valid_langs or args.tgt_lang not in valid_langs:
        parser.error(
            f"Invalid language code. Supported: {', '.join(lang.value for lang in valid_langs)}"
        )

    # Validate file paths
    if args.input_file and not Path(args.input_file).is_file():
        parser.error(f"Input file {args.input_file} does not exist")

    return args


def initialize_logger(config: ConfigManager) -> LoggerType:
    log_file, name = config.logging_config.translate

    return AppLogger(name, log_file).get_logger


def main():
    args = parse_args()

    # Initialize configurations
    config_manager = ConfigManager(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        input_file=args.input_file,
    )

    logger = initialize_logger(config_manager)

    try:
        # Step 1: Check for GPU availability
        if not torch.cuda.is_available():
            logger.warning("⚠️ No GPU detected, translation will be slow!")

        logger.debug(f"CUDA version: {torch.version.cuda}")  # type: ignore

        # Step 2: Start translation
        pipeline = TranslationPipeline(config_manager, logger)
        pipeline.run_translation(
            translated_file_name=args.translated_file_name,
            model_cli_path=args.model_path,
        )
    except Exception as e:
        logger.error(f"🛑 An error occurred during translation: {e}")
        raise


if __name__ == "__main__":
    main()
