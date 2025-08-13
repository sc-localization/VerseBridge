import argparse
from pathlib import Path
from src.config import ConfigManager
from src.ner import NERPipeline
from src.utils import AppLogger, SystemUtils, HelpUtils, CustomHelpFormatter
from src.type_defs import LoggerType


def parse_args():
    logger = AppLogger("cli_parser").get_logger
    system_utils = SystemUtils()
    help_utils = HelpUtils()

    system_language = system_utils.get_system_lang_code()
    help_strings = help_utils.get_help_strings(
        "help_strings_ner.json", system_language, logger
    )

    parser = argparse.ArgumentParser(
        description=help_strings["description"],
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        metavar="",
        help=help_strings["input_file_help"],
    )
    parser.add_argument(
        "--stage",
        choices=["extract", "review", "bio", "train"],
        required=True,
        help=help_strings["stage_help"],
    )

    args = parser.parse_args()

    # Validate file paths
    if args.input_file and not Path(args.input_file).is_file():
        parser.error(f"Input file {args.input_file} does not exist")

    return args


def initialize_logger(config: ConfigManager) -> LoggerType:
    log_file, name = config.logging_config.ner

    return AppLogger(name, log_file).get_logger


def main():
    args = parse_args()

    # Initialize configurations
    config_manager = ConfigManager(input_file=args.input_file)
    logger = initialize_logger(config_manager)

    logger.info(f"Stage: {args.stage}")
    pipeline = NERPipeline(config_manager, logger)

    try:
        if args.stage == "extract":
            pipeline.run_extraction()
        elif args.stage == "review":
            pipeline.run_review()
        elif args.stage == "bio":
            pipeline.run_bio_conversion()
        elif args.stage == "train":
            pipeline.run_training()
    except Exception as e:
        logger.error(
            f"🛑 An error occurred during ner pipeline: {str(e)}"
        )
        raise


if __name__ == "__main__":
    main()
