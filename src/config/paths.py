from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, cast

from src.type_defs import IniFilePathsType, JsonFilePathsType


@dataclass
class PathConfig:
    input_file_path: str | None = None
    original_ini_file: str = "global_original.ini"
    translated_ini_file: str = "global_pre_translated.ini"

    base_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    translation_dir: Path = field(init=False)
    logging_dir: Path = field(init=False)
    ini_files: IniFilePathsType = field(init=False)
    json_files: JsonFilePathsType = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(__file__).parent.parent.parent
        self.base_dir = self.base_dir.relative_to(Path.cwd())
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.translation_dir = self.data_dir / "translation_results"
        self.logging_dir = self.base_dir / "logs"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.ini_files = IniFilePathsType(
            original=(
                Path(self.input_file_path)
                if self.input_file_path
                else self.data_dir / self.original_ini_file
            ),
            translated=self.data_dir / self.translated_ini_file,
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
        ini_values: List[Path] = list(cast(Dict[str, Path], self.ini_files).values())
        missing_files: List[Path] = [path for path in ini_values if not path.exists()]

        if missing_files:
            raise FileNotFoundError(
                f"The following necessary INI files are missing: {', '.join(map(str, missing_files))}"
            )
