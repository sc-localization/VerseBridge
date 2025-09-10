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
from transformers import (
    PreTrainedModel,
    TFPreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from peft import PeftModel, PeftMixedModel


LoggerType: TypeAlias = logging.Logger
ArgLoggerType: TypeAlias = Optional[LoggerType]


class LangCode(str, Enum):
    EN = "en"
    RU = "ru"


class MappedCode(str, Enum):
    ENG_LATN = f"<2{LangCode.EN.value}>"
    RUS_CYRL = f"<2{LangCode.RU.value}>"


LangMapType = Dict[LangCode, MappedCode]


class IniFilePathsType(TypedDict):
    original: Path  # Path to the source INI file for model training
    translated: Path  # Path to the pre-translated INI file for training


class JsonTrainedDataFilePathsType(TypedDict):
    train: Path
    test: Path


class LogConfigType(NamedTuple):
    log_file: str
    logger_name: str


GenerationConfigType: TypeAlias = Dict[str, Any]


class TokenizerOptionsType(TypedDict):
    return_tensors: Literal["pt"]
    padding: bool
    truncation: bool


class TokenizerConfigType(TokenizerOptionsType):
    max_length: int


class Optimizer(str, Enum):
    adafactor = "adafactor"
    adamw_torch = "adamw_torch"
    paged_adamw_8bit = "adamw_bnb_8bit"
    paged_adamw_32bit = "paged_adamw_32bit"


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

HelpStringsDictType: TypeAlias = Dict[str, str]


class HelpStringsLangType(TypedDict):
    ru: HelpStringsDictType
    en: HelpStringsDictType


class JSONDataTranslationType(TypedDict):
    original: INIFIleValueType
    translated: INIFIleValueType


class EntityType(TypedDict):
    start: int
    end: int
    label: str


EntitiesType: TypeAlias = List[EntityType]


class JSONDataNERType(TypedDict):
    id: INIFIleKeyType
    text: INIFIleValueType
    entities: EntitiesType


class JSONConvertedToBIOType(TypedDict):
    id: INIFIleKeyType
    tokens: List[str]
    labels: List[str]


JSONHelpStringsDictType = HelpStringsLangType
JSONDataTranslationListType: TypeAlias = List[JSONDataTranslationType]
JSONDataNERListType: TypeAlias = List[JSONDataNERType]
JSONDataConvertedToBIOListType: TypeAlias = List[JSONConvertedToBIOType]
JSONNERListType: TypeAlias = List[str]


LoadedJSONType: TypeAlias = (
    JSONHelpStringsDictType
    | JSONDataTranslationListType
    | JSONDataNERListType
    | JSONDataConvertedToBIOListType
    | JSONNERListType
)

CleanedINIFIleValueType: TypeAlias = INIFIleValueType


TranslationModelNameType: TypeAlias = Literal["google/madlad400-3b-mt"]
NerModelNameType: TypeAlias = Literal["Jean-Baptiste/roberta-large-ner-english"]
ModelPathType: TypeAlias = str
ModelCLIType: TypeAlias = Optional[ModelPathType]
ModelPathOrName: TypeAlias = TranslationModelNameType | NerModelNameType | ModelPathType


LastCheckpointPathType: TypeAlias = Optional[Path]
InitializedModelType = PeftModel | PeftMixedModel | PreTrainedModel | TFPreTrainedModel
InitializedTokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast
TranslatedFileNameType = Optional[str]

BufferType: TypeAlias = List[TranslatedIniLineType]

TranslatorCallableType: TypeAlias = Callable[
    [List[INIFIleValueType]], List[TranslatedIniValueType]
]

AppTaskType: TypeAlias = Literal["ner", "translation"]  # TODO: use enum if posible
CharToTokenType: TypeAlias = List[Tuple[int, int]]
AggregationStrategyType: TypeAlias = Literal["simple", "average", "max", "none"]


class CachedParamsType(TypedDict):
    max_model_length: int
    token_reserve: int


class TranslationTrainingConfigType(TypedDict):
    logging_dir: str

    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float

    optim: Optimizer
    lr_scheduler_type: Scheduler

    bf16: bool
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


class NerTrainingConfigType(TypedDict):
    logging_dir: str

    num_train_epochs: int
    learning_rate: float
    weight_decay: float

    per_device_train_batch_size: int
    per_device_eval_batch_size: int

    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    evaluation_strategy: Strategy


LoraTargetModulesType: TypeAlias = Optional[Union[list[str], str]]


class NerLabelConfig(TypedDict):
    num_labels: int
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    ignore_mismatched_sizes: bool


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


def is_json_data_tranlation_list_type(
    data: Any,
) -> TypeGuard[JSONDataTranslationListType]:
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


def is_json_data_ner_list_type(data: Any) -> TypeGuard[JSONDataNERListType]:
    if not isinstance(data, list):
        return False

    for item in data:
        if not isinstance(item, dict):
            return False

        if "id" not in item or "text" not in item or "entities" not in item:
            return False

        if not isinstance(item["id"], str) or not isinstance(item["text"], str):
            return False

        if not isinstance(item["entities"], list):
            return False

        for entity in item["entities"]:
            if not isinstance(entity, dict):
                return False

            if "start" not in entity or "end" not in entity or "label" not in entity:
                return False

            if not isinstance(entity["start"], int) or not isinstance(
                entity["end"], int
            ):
                return False
            if not isinstance(entity["label"], str):
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


def is_list_of_ner_patterns(obj: Any) -> TypeGuard[JSONNERListType]:
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)
