from dataclasses import dataclass

from src.type_defs import LogConfigType


@dataclass(frozen=True)
class LoggingConfig:
    translate: LogConfigType = LogConfigType("translate.log", "translate_logger")
    train: LogConfigType = LogConfigType("train.log", "train_logger")
    preprocess: LogConfigType = LogConfigType("preprocess.log", "preprocess_logger")
    default: LogConfigType = LogConfigType("default.log", "default_logger")
    ner: LogConfigType = LogConfigType("ner.log", "ner_logger")
