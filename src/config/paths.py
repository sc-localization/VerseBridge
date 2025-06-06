from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.type_defs import IniFilePathsType, JsonFilePathsType


@dataclass
class BasePathConfig:
    default_source_file: str = "global_original.ini"
    pre_translated_file: str = "global_pre_translated.ini"

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


@dataclass
class TranslationPathConfig(BasePathConfig):
    input_file: Optional[str] = None
    translation_dir: Path = field(init=False)
    input_file_path: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.translation_dir = self.data_dir / "translation_results"

        # Initialize input_file_path
        if self.input_file is None:
            self.input_file_path = self.data_dir / self.default_source_file
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
    ini_files: IniFilePathsType = field(init=False)
    json_files: JsonFilePathsType = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.ini_files = IniFilePathsType(
            original=self.data_dir / self.default_source_file,
            translated=self.data_dir / self.pre_translated_file,
        )

        self.json_files = JsonFilePathsType(
            data=self.data_dir / "data.json",
            cleaned=self.data_dir / "cleaned_data.json",
            train=self.data_dir / "train.json",
            test=self.data_dir / "test.json",
        )

    def data_files_exist(self) -> bool:
        return self.json_files["train"].exists() and self.json_files["test"].exists()

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
