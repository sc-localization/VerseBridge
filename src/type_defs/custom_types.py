from enum import Enum
import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeAlias,
    TypeGuard,
    TypedDict,
    Union,
)
from transformers import PreTrainedModel
from peft import PeftModel, PeftMixedModel


LoggerType: TypeAlias = logging.Logger
ArgLoggerType: TypeAlias = Optional[LoggerType]


class LangCode(str, Enum):
    EN = "en"
    RU = "ru"


class MappedCode(str, Enum):
    ENG_LATN = "eng_Latn"
    RUS_CYRL = "rus_Cyrl"


LangMapType = Dict[LangCode, MappedCode]


class IniFilePathsType(TypedDict):
    original: Path
    translated: Path


class JsonFilePathsType(TypedDict):
    data: Path
    cleaned: Path
    train: Path
    test: Path


class LogConfigType(NamedTuple):
    log_file: str
    logger_name: str


class GenerationConfigType(TypedDict):
    num_beams: int
    top_k: int
    top_p: float
    do_sample: bool


class TokenizerConfigType(TypedDict):
    return_tensors: Literal["pt"]
    padding: bool
    truncation: bool
    max_length: int


class Optimizer(str, Enum):
    adafactor = "adafactor"
    adamw = "adamw"
    adamw_torch = "adamw_torch"


class Scheduler(str, Enum):
    cosine = "cosine"
    linear = "linear"
    polynomial = "polynomial"
    constant = "constant"


class Strategy(str, Enum):
    STEPS = "steps"
    EPOCH = "epoch"


class Metric(str, Enum):
    BERTSCORE_F1 = "bertscore_f1"
    BLEU = "bleu"
    ROUGE = "rouge"
    LOSS = "loss"


class LogReportTarget(str, Enum):
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    NONE = "none"


LogReportTargetListType: TypeAlias = List[LogReportTarget]


class LabelNames(str, Enum):
    LABELS = "labels"


LabelNamesListType: TypeAlias = List[Literal[LabelNames.LABELS]]

MetricScoresType: TypeAlias = Dict[
    str, float
]  # bleu: float, chrf: float, meteor: float, bertscore_f1: float

ExcludeKeysType: TypeAlias = Tuple[str, ...]
ProtectedPatternsType: TypeAlias = Tuple[str, ...]
PlaceholdersType: TypeAlias = Dict[str, str]


INIFIleKeyType: TypeAlias = str
INIFIleValueType: TypeAlias = str
KeyValuePairType: TypeAlias = Tuple[INIFIleKeyType, INIFIleValueType]
IniLineType: TypeAlias = Literal[f"{INIFIleKeyType}={INIFIleValueType}"]
TranslatedIniValueType: TypeAlias = INIFIleValueType
TranslatedIniLineType: TypeAlias = Literal[
    f"{INIFIleKeyType}={INIFIleValueType|TranslatedIniValueType}"
]

INIDataType: TypeAlias = Dict[INIFIleKeyType, INIFIleValueType]

IniFileListLinesType: TypeAlias = List[KeyValuePairType]

HelpStringsDictType: TypeAlias = Dict[str, str]


class HelpStringsLangType(TypedDict):
    ru: HelpStringsDictType
    en: HelpStringsDictType


class JSONDataType(TypedDict):
    original: INIFIleValueType
    translated: INIFIleValueType


JSONHelpStringsDictType = HelpStringsLangType
JSONDataListType: TypeAlias = List[JSONDataType]

LoadedJSONType: TypeAlias = JSONHelpStringsDictType | JSONDataListType

CleanedINIFIleValueType: TypeAlias = INIFIleValueType


ModelNameType: TypeAlias = Literal["facebook/nllb-200-distilled-1.3B"]
ModelPathType: TypeAlias = str
ModelCLIType: TypeAlias = Optional[ModelPathType]
ModelPathOrName: TypeAlias = ModelNameType | ModelPathType


LastCheckpointPathType: TypeAlias = Optional[Path]
InitializedModelType = PeftModel | PeftMixedModel | PreTrainedModel
TranslatedFileNameType = Optional[str]

BufferType: TypeAlias = List[TranslatedIniLineType]

TranslatorCallableType: TypeAlias = Callable[[str], str]

TranslationPriorityType = Literal["output", "existing"]


class TrainingConfigType(TypedDict):
    logging_dir: str

    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float

    optim: Optimizer
    lr_scheduler_type: Scheduler

    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    eval_accumulation_steps: int

    eval_strategy: Strategy
    eval_steps: int
    eval_on_start: bool
    predict_with_generate: bool
    save_strategy: Strategy
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: Metric
    greater_is_better: bool

    logging_steps: int
    logging_strategy: Strategy
    report_to: LogReportTargetListType

    label_smoothing_factor: float
    label_names: LabelNamesListType

    torch_empty_cache_steps: int


LoraTargetModulesType: TypeAlias = Optional[Union[list[str], str]]


def is_json_help_strings_dict_type(data: Any) -> TypeGuard[JSONHelpStringsDictType]:
    if not isinstance(data, dict):
        return False

    if "ru" not in data or "en" not in data:
        return False

    if not isinstance(data["ru"], dict) or not isinstance(data["en"], dict):
        return False

    if not all(isinstance(v, str) for v in data["ru"].values()):
        return False

    if not all(isinstance(v, str) for v in data["en"].values()):
        return False

    return True


def is_json_data_list_type(data: Any) -> TypeGuard[JSONDataListType]:
    if not isinstance(data, list):
        return False

    for item in data:
        if not isinstance(item, dict):
            return False

        if "original" not in item or "translated" not in item:
            return False

        if not isinstance(item["original"], str) or not isinstance(
            item["translated"], str
        ):
            return False

    return True


def is_ini_file_line(line: Any) -> TypeGuard[IniLineType]:
    if not isinstance(line, str):
        return False

    if "=" not in line:
        return False

    key, _ = line.split("=", 1)

    if not key:  # line must be with empty value
        return False

    return True
