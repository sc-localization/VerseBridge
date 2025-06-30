import argparse
import torch


from src.config import ConfigManager, LanguageConfig
from src.utils import AppLogger, CustomHelpFormatter, SystemUtils, HelpUtils
from src.preprocess import PreprocessPipeline
from src.training import TrainingPipeline
from src.type_defs import (
    LangCode,
    LoggerType,
)


def needs_preprocessing(config: ConfigManager, logger: LoggerType) -> bool:
    data_files_exist = config.training_path_config.data_files_exist()

    if data_files_exist:
        logger.info("🔍 Train and test files found, skipping preprocessing")
        return False

    logger.info("🔃 Train/test files missing, preprocessing required")
    return True


def parse_args():
    logger = AppLogger("cli_parser").get_logger
    system_utils = SystemUtils()
    help_utils = HelpUtils()

    system_language = system_utils.get_system_lang_code()
    help_strings = help_utils.get_help_strings(
        "help_strings_train.json", system_language, logger
    )

    parser = argparse.ArgumentParser(
        description=help_strings["description"],
        formatter_class=CustomHelpFormatter,
        epilog=help_strings["epilog"],
    )
    parser.add_argument(
        "--src-lang",
        default=LangCode.EN,
        type=LangCode,
        choices=LangCode,
        metavar="",
        help=help_strings["src_lang_help"],
    )
    parser.add_argument(
        "--tgt-lang",
        default=LangCode.RU,
        type=LangCode,
        choices=LangCode,
        metavar="",
        help=help_strings["tgt_lang_help"],
    )
    parser.add_argument(
        "--with-lora",
        action="store_true",
        help=help_strings["with_lora_help"],
    )
    parser.add_argument(
        "--model-path",
        default=None,
        type=str,
        metavar="PATH",
        help=help_strings["model_path_help"],
    )

    args = parser.parse_args()

    valid_langs = LanguageConfig().available_languages

    if args.src_lang not in valid_langs or args.tgt_lang not in valid_langs:
        parser.error(
            f"Invalid language code. Supported: {', '.join(lang.value for lang in valid_langs)}"
        )

    return args


def initialize_logger(config: ConfigManager) -> LoggerType:
    log_file, name = config.logging_config.train

    return AppLogger(
        name,
        log_file,
        log_dir=config.base_path_config.logging_dir,
    ).get_logger


def main():
    args = parse_args()

    config_manager = ConfigManager(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )
    logger = initialize_logger(config_manager)

    try:
        # Step 1: Preprocessing (if necessary)
        if needs_preprocessing(config_manager, logger):
            logger.info("⚙️ Running data preprocessing...")
            preprocessor = PreprocessPipeline(config_manager, logger)
            preprocessor.run_preprocess()

        # Step 2: Check for GPU availability
        if not torch.cuda.is_available():
            logger.warning("⚠️ No GPU detected, training will be slow!")

        logger.debug(f"CUDA version: {torch.version.cuda}")  # type: ignore

        # Step 3: Start training
        pipeline = TrainingPipeline(config_manager, logger)
        pipeline.run_training(model_cli_path=args.model_path, with_lora=args.with_lora)
    except Exception as e:
        logger.error(f"🛑 An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
