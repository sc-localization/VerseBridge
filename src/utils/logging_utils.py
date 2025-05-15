from pathlib import Path


class LoggingUtils:
    @staticmethod
    def sanitize_path_for_log(path: str) -> str:
        """Sanitizes a path for safe logging, returning only the file/directory name if the path exists."""
        if not path:
            return "<invalid_path>"
        try:
            path_obj = Path(path)

            if path_obj.exists():
                return path_obj.name

            return path
        except (ValueError, OSError):
            return path
