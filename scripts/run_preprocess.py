from src.config import ConfigManager
from src.preprocess import PreprocessPipeline
from src.utils import AppLogger
from src.type_defs import LoggerType


def initialize_logger(config: ConfigManager) -> LoggerType:
    log_file, name = config.logging_config.preprocess

    return AppLogger(name, log_file).get_logger


def main():
    config_manager = ConfigManager()
    logger = initialize_logger(config_manager)

    try:
        pipeline = PreprocessPipeline(config_manager, logger)
        pipeline.run_preprocess()
    except Exception as e:
        logger.error(f"🛑 An error occurred during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
