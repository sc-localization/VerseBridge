import re
import pytest
from src.config import ConfigManager


@pytest.fixture
def protected_patterns():
    """Возвращает protected_patterns из TranslationConfig."""
    return ConfigManager().translation_config.protected_patterns


@pytest.mark.parametrize(
    "pattern, test_string, should_match",
    [
        (0, "%X23", True),  # Matches variables like %X23
        (0, "random text", False),  # Does not match random text
        (1, "#~action(param)", True),  # Matches #~action(...)
        (1, "#mission(param)", False),  # Does not match #~mission(...)
        (1, "~action(param)", False),  # Does not match without #
        (2, "~action(param)", True),  # Matches ~action(...)
        (2, "action(param)", False),  # Does not match without ~
        (2, "~mission(param)", True),  # Matches ~mission(...)
        (3, "~action", True),  # Matches ~action
        (3, "mission", False),  # Does not match without ~
        (4, "aUEC", True),  # Matches currency aUEC
        (4, "random text", False),  # Does not match random text
        (5, "[example]", True),  # Matches text in square brackets
        (5, "example", False),  # Does not match without brackets
        (6, "<tag>", True),  # Matches opening or closing tags
        (6, "tag", False),  # Does not match without <>
        (7, "</tag>", True),  # Matches closing tags
        (7, "tag", False),  # Does not match without </>
        (8, "<tag>", True),  # Matches opening tags without /
        (8, "</tag>", False),  # Does not match closing tags
    ],
)
def test_no_translate_patterns(protected_patterns, pattern, test_string, should_match):
    match = re.search(protected_patterns[pattern], test_string)
    assert bool(match) == should_match
