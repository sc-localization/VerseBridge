import logging
from pathlib import Path

from src.type_defs import BufferType


class BufferedFileWriter:
    def __init__(self, file_path: Path, buffer_size: int, logger: logging.Logger):
        """
        Initializes BufferedFileWriter.

        Args:
            file_path: Path to the file for writing.
            buffer_size: Buffer size (number of lines before writing).
            logger: Logger for logging operations.
        """
        self.logger = logger
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.buffer: BufferType = []
        self.file = None

    def __enter__(self):
        self.file = self.file_path.open("w", encoding="utf-8-sig")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        self.flush()
        if self.file:
            self.file.close()

    def write(self, line: str) -> None:
        """
        Adds a line to the buffer and flushes it if full.

        Args:
            line: Line to write.
        """
        self.buffer.append(line)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """
        Writes the contents of the buffer to the file and clears the buffer.
        """
        if self.buffer and self.file:
            try:
                self.file.writelines(self.buffer)
                self.buffer.clear()
            except Exception as e:
                self.logger.error(f"Failed to write to {self.file_path}: {str(e)}")
                raise
