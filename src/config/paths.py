from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.type_defs import (
    IniFilePathsType,
    LangCode,
    JsonTrainedDataFilePathsType,
)


@dataclass
class BasePathConfig:
    default_source_file: str = field(default="global_original.ini", init=False)
    pre_translated_file: str = field(default="global_pre_translated.ini", init=False)

    base_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    logging_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(__file__).parent.parent.parent
        self.base_dir = self.base_dir.relative_to(Path.cwd())
        self.data_dir = self.base_dir / "data"

        self.models_dir = self.base_dir / "models"
        self.logging_dir = self.base_dir / "logs"

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_raw_data_dir_by_lang(self, lang: LangCode) -> Path:
        raw_dir = self.data_dir / "raw" / lang.value
        raw_dir.mkdir(parents=True, exist_ok=True)

        return raw_dir

    def get_training_data_dir_by_lang(
        self, src_lang: LangCode, tgt_lang: LangCode
    ) -> Path:
        training_dir = self.data_dir / "training" / f"{src_lang.value}-{tgt_lang.value}"
        training_dir.mkdir(parents=True, exist_ok=True)

        return training_dir

    def get_translation_results_dir_by_lang(
        self, src_lang: LangCode, tgt_lang: LangCode
    ) -> Path:
        translation_dir = (
            self.data_dir / "translation_results" / f"{src_lang.value}-{tgt_lang.value}"
        )
        translation_dir.mkdir(parents=True, exist_ok=True)

        return translation_dir

    def get_ner_data_dir(self) -> Path:
        ner_dir = self.data_dir / "ner"
        ner_dir.mkdir(parents=True, exist_ok=True)

        return ner_dir


@dataclass
class TranslationPathConfig(BasePathConfig):
    src_lang: LangCode = LangCode.EN
    tgt_lang: LangCode = LangCode.RU
    input_file: Optional[str] = None
    translation_dir: Path = field(init=False)
    input_file_path: Path = field(init=False)
    translation_src_dir: Path = field(init=False)
    translation_dest_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.translation_dir = self.get_translation_results_dir_by_lang(
            self.src_lang, self.tgt_lang
        )
        self.translation_src_dir = self.translation_dir / self.src_lang.value
        self.translation_dest_dir = self.translation_dir / self.tgt_lang.value

        # Initialize input_file_path
        if self.input_file is None:
            self.input_file_path = (
                self.get_raw_data_dir_by_lang(self.src_lang) / self.default_source_file
            )
        else:
            self.input_file_path = Path(self.input_file)

            if not self.input_file_path.exists():
                raise FileNotFoundError(
                    f"Input file does not exist: {self.input_file_path}"
                )

    def check_input_file_exists(self) -> None:
        if not self.input_file_path.exists():
            raise FileNotFoundError(
                f"The input INI file does not exist: {self.input_file_path}"
            )


@dataclass
class TrainingPathConfig(BasePathConfig):
    src_lang: LangCode = LangCode.EN
    tgt_lang: LangCode = LangCode.RU

    raw_dataset: Path = field(init=False)
    cleaned_dataset: Path = field(init=False)

    ini_files: IniFilePathsType = field(init=False)
    trained_data_files: JsonTrainedDataFilePathsType = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        training_dir = self.get_training_data_dir_by_lang(self.src_lang, self.tgt_lang)

        self.ini_files = IniFilePathsType(
            original=self.get_raw_data_dir_by_lang(self.src_lang)
            / self.default_source_file,
            translated=self.get_raw_data_dir_by_lang(self.tgt_lang)
            / self.pre_translated_file,
        )

        self.raw_dataset = training_dir / "data.json"
        self.cleaned_dataset = training_dir / "cleaned_data.json"

        self.trained_data_files = JsonTrainedDataFilePathsType(
            train=training_dir / "train.json",
            test=training_dir / "test.json",
        )

    def data_files_exist(self) -> bool:
        return (
            self.trained_data_files["train"].exists()
            and self.trained_data_files["test"].exists()
        )

    def check_ini_files_exist(self) -> None:
        ini_values: List[Path] = [
            self.ini_files["original"],
            self.ini_files["translated"],
        ]
        missing_files: List[Path] = [path for path in ini_values if not path.exists()]

        if missing_files:
            raise FileNotFoundError(
                f"The following necessary INI files are missing: {', '.join(map(str, missing_files))}"
            )


@dataclass
class NERPathConfig(BasePathConfig):
    input_file: Optional[str] = None

    input_file_path: Path = field(init=False)

    extracted_ner_data_path: Path = field(init=False)

    temp_streamlit_data_path: Path = field(init=False)
    corrected_streamlit_data_path: Path = field(init=False)

    bio_ner_dataset_path: Path = field(init=False)
    trained_data_files: JsonTrainedDataFilePathsType = field(init=False)

    output_dir: Path = field(init=False)
    logging_dir: Path = field(init=False)

    ner_patterns_path: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        raw_dir = self.get_raw_data_dir_by_lang(
            LangCode.EN
        )  # always use original English file
        ner_data_dir = self.get_ner_data_dir()

        self.extracted_ner_data_path = ner_data_dir / "ner_unannotated.json"

        self.temp_streamlit_data_path = ner_data_dir / "temp_corrected.json"
        self.corrected_streamlit_data_path = ner_data_dir / "dataset_corrected.json"

        self.bio_ner_dataset_path = ner_data_dir / "dataset_bio.json"

        self.trained_data_files = JsonTrainedDataFilePathsType(
            train=ner_data_dir / "train.json",
            test=ner_data_dir / "test.json",
        )

        self.output_dir = self.models_dir / "ner_output"
        self.logging_dir = self.logging_dir / "ner_logs"

        self.ner_patterns_path = ner_data_dir / "ner_patterns.json"

        # Initialize input_file_path
        if self.input_file is None:
            self.input_file_path = raw_dir / self.default_source_file
        else:
            self.input_file_path = Path(self.input_file)

            if not self.input_file_path.exists():
                raise FileNotFoundError(
                    f"Input file does not exist: {self.input_file_path}"
                )

    def check_input_file_exists(self) -> None:
        if not self.input_file_path.exists():
            raise FileNotFoundError(
                f"The input INI file does not exist: {self.input_file_path}"
            )

    def data_files_exist(self) -> bool:
        return (
            self.trained_data_files["train"].exists()
            and self.trained_data_files["test"].exists()
        )
