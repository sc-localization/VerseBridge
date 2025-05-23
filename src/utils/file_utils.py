import json
import shutil
from pathlib import Path
from typing import Union

from src.type_defs import (
    ExcludeKeysType,
    ArgLoggerType,
    INIDataType,
    JSONDataListType,
    LoadedJSONType,
    KeyValuePairType,
    IniLineType,
    is_ini_file_line,
)
from .app_logger import AppLogger


class FileUtils:
    def __init__(self, logger: ArgLoggerType = None):
        self.logger = logger or AppLogger("file_utils").get_logger

    def _parse_line(self, line: IniLineType) -> KeyValuePairType | None:
        key_value = line.split("=", 1)

        if len(key_value) == 2:
            key, value = key_value

            return key.strip(), value.strip()

    def parse_ini_file(
        self, file_path: Path, exclude_keys: ExcludeKeysType = tuple()
    ) -> INIDataType:
        """
        Reads an INI file and returns a dictionary with key-value pairs.

        Args:
            file_path: Path to the INI file.
            exclude_keys: Set of keys to exclude (default is None for translation).

        Returns:
            Dictionary with key-value pairs from the INI file.
        """
        data: INIDataType = {}
        file_path = Path(file_path)

        self.logger.debug(f"Parsing INI file: {file_path}")

        try:
            with file_path.open(encoding="utf-8-sig") as file:
                for line in file:
                    line = line.strip()

                    if not line or line.startswith(";"):
                        continue

                    if is_ini_file_line(line):
                        key_value = self._parse_line(line)
                    else:
                        self.logger.warning(f"Invalid line format: {line}")
                        continue

                    if key_value is None:
                        continue

                    key, value = key_value

                    if key in exclude_keys:
                        continue

                    if not value:
                        self.logger.debug(f"Parsed empty value for key {key}")

                    data[key] = value

            self.logger.debug(f"Parsed {len(data)} key-value pairs from {file_path}")

            return data
        except Exception as e:
            self.logger.error(f"Failed to parse INI file {file_path}: {str(e)}")
            raise

    def copy_files(
        self,
        src_path: Union[str, Path],
        dest_path: Union[str, Path],
        dest_name: str | None = None,
    ) -> None:
        """
        Copies files or directories from src_path to dest_path.

        Args:
            src_path: Path to the source file or directory.
            dest_path: Path to the target directory.
        """
        src_path = Path(src_path)
        dest_path = Path(dest_path)

        self.logger.debug(f"Copying files from {src_path} to {dest_path}...")

        try:
            if dest_path.exists() and not dest_path.is_dir():
                self.logger.error(
                    f"Destination path {dest_path} is a file, not a directory"
                )
                raise ValueError(f"Destination path {dest_path} must be a directory")

            if dest_path.exists():
                self.logger.debug(f"Clearing existing directory: {dest_path}")
                shutil.rmtree(dest_path)

            dest_path.mkdir(parents=True, exist_ok=True)

            if src_path.is_file():
                dest_file = dest_path / (dest_name or src_path.name)

                self._copy_file(src_path, dest_file)
            elif src_path.is_dir():
                self._copy_directory(src_path, dest_path)
            else:
                self.logger.error(f"Source path {src_path} is invalid")
                raise FileNotFoundError(f"Source path {src_path} is invalid")

        except Exception as e:
            self.logger.error(
                f"Failed to copy from {src_path} to {dest_path}: {str(e)}"
            )
            raise

    def _copy_file(self, src_file: Path, dest_file: Path) -> None:
        """
        Copies a single file from src_file to dest_file.

        Args:
            src_file: Path to the source file.
            dest_file: Path to the target file.
        """
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_file, dest_file)

        self.logger.debug(f"Copied file: {src_file} to {dest_file}")

    def _copy_directory(self, src_dir: Path, dest_dir: Path) -> None:
        """
        Copies a directory and its contents from src_dir to dest_dir.

        Args:
            src_dir: Source directory.
            dest_dir: Destination directory.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Copying directory: {src_dir} to {dest_dir}")

        for item in src_dir.iterdir():
            src_item = src_dir / item.name
            dest_item = dest_dir / item.name

            if src_item.is_file():
                self._copy_file(src_item, dest_item)
            elif src_item.is_dir():
                self._copy_directory(src_item, dest_item)

    def load_json(self, file_path: Path) -> LoadedJSONType:
        """
        Loads JSON data from a file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Data from the JSON file (list, dict, or other structures).
        """
        self.logger.debug(f"Loading JSON data from {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data: LoadedJSONType = json.load(file)

            self.logger.info(f"Loaded JSON from {file_path}")

            return data
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {str(e)}")
            raise

    def save_json(
        self,
        data: JSONDataListType,
        output_file: Path,
    ) -> None:
        """
        Saves data to a JSON file.

        Args:
            data: Data to save (list of dictionaries).
            output_file: Path to the output JSON file.
        """

        self.logger.debug(f"Saving data to JSON file: {output_file}")

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with output_file.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)

            self.logger.info(f"Saved JSON to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {output_file}: {str(e)}")
            raise
