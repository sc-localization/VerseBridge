import argparse
from typing import List


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter for improved --help output with support for \n in help."""

    def __init__(self, prog: str) -> None:
        """
        Custom formatter for improved --help output with support for \n in help.

        Args:
            prog (str): The name of the program.
        """
        super().__init__(prog)
        self._max_help_position = 40  # Column width for parameter names

    def _split_lines(self, text: str, width: int) -> List[str]:
        """
        Split the given text into lines with a maximum width.

        Args:
            text (str): The text to split.
            width (int): The maximum width of each line.

        Returns:
            List[str]: A list of lines.
        """
        lines: List[str] = []

        for line in text.split("\n"):
            current_line = ""

            for word in line.split():
                if len(current_line) + len(word) + 1 <= width:
                    current_line += word + " "
                else:
                    if current_line:
                        lines.append(current_line.rstrip())

                    current_line = word + " "

            if current_line:
                lines.append(current_line.rstrip())

        return lines
