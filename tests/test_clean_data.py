import pytest
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from src.config import ConfigManager
from src.preprocess.text_cleaner import TextCleaner


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")


@pytest.fixture
def text_cleaner(tokenizer):
    return TextCleaner(
        tokenizer=tokenizer,
        protected_patterns=ConfigManager().translation_config.protected_patterns,
        max_model_length=512,
        logger=MagicMock(),
    )


@pytest.mark.parametrize(
    "input_text, remove_patterns, expected",
    [
        # Basic cases without removing patterns
        ("Welcome to ~mission(Location)", False, "Welcome to ~mission(Location)"),
        ("Press #~action(Button)", False, "Press #~action(Button)"),
        # Removing patterns
        ("Welcome to ~mission(Location)", True, "Welcome to"),
        ("Press #~action(Button)", True, "Press"),
        # Processing line breaks
        ("Line1\\nLine2", False, "Line Line"),
        ("Line 3\\nLine 4", False, "Line 3 Line 4"),
        ("Double\\\\n", False, "Double"),
        # Clearing spaces
        ("   Hello   ", False, "Hello"),
        ("Line  \n  next", False, "Line next"),
        ("Release \nRelease", False, "Release Release"),
        # HTML entities
        ("<>&amp;", False, ""),
        ("test&nbsp;test", False, "test test"),
        # Symbols and punctuation
        ("“quotes”", False, '"quotes"'),
        ("Mixed… dots…", False, "Mixed... dots..."),
        ("**bold**", False, "bold"),
        ("Long—dash", False, "Long-dash"),
        ("Double-—dash", False, "Double-dash"),
        ("Symbol -------line____ ======Test", False, "Symbol line Test"),
        ("End of sentence:", False, "End of sentence:"),
        ("End of sentence?", False, "End of sentence?"),
        ("End of sentence: /", False, "End of sentence:"),
        ("Space , after , symbols", False, "Space after symbols"),
        (
            "(Curly) brackets at the beginning",
            False,
            "(Curly) brackets at the beginning",
        ),
        (
            "() Space brackets at the beginning",
            False,
            "Space brackets at the beginning",
        ),
        ("Brackets at the end (Curly)", False, "Brackets at the end (Curly)"),
        ("Space brackets at the end ()", False, "Space brackets at the end"),
        (
            "Any #: strange (symbols) %^& between ;% :! words!",
            False,
            "Any strange (symbols) between words!",
        ),
        # Edge cases
        ("", False, ""),
        (None, False, ""),
        ("\n\n\n", False, ""),
        ("   ", False, ""),
    ],
)
def test_clean_text(text_cleaner, input_text, remove_patterns, expected):
    assert text_cleaner.clean_text(input_text, remove_patterns) == expected


@pytest.mark.parametrize(
    "input_text, remove_patterns, expected",
    [
        (
            "%REMOVE%\n~action()\n<div>Hello&nbsp;world…</div>\n\n\nPrice: 100 aUEC\n-----",
            False,
            "%REMOVE <div>Hello world...</div> Price: 100 aUEC",
        ),
        (
            "%REMOVE%\n~action()\n<div>Hello&nbsp;world…</div>\n\n\nPrice: 100aUEC\n-----",
            True,
            "Hello world... Price: 100",
        ),
        (
            "Mixed content: [not-ignore] %var \nText «quotes» ---",
            False,
            'Mixed content: [not-ignore] %var Text "quotes"',
        ),
        (
            "Mixed content: [not-ignore] %var \nText «quotes» ---",
            True,
            'Mixed content: Text "quotes"',
        ),
        (
            "!@# %( *(Location) ()!@(Option) !@ # hello? ##~action() #: world... $% ! test, #$ $%!!(curly) dots...symbol?",
            False,
            "(Location) (Option) hello? world... test, (curly) dots...symbol?",
        ),
        (
            "!@# %( *(Location) ()!@(Option) !@ # hello? ##~action() #: world... $% ! test, #$ $%!!(curly) dots...symbol?",
            True,
            "(Location) (Option) hello? world... test, (curly) dots...symbol?",
        ),
    ],
)
def test_clean_text_complex(text_cleaner, input_text, remove_patterns, expected):
    assert (
        text_cleaner.clean_text(input_text, remove_patterns).strip() == expected.strip()
    )


@pytest.mark.parametrize(
    "source_text, target_text, expected",
    [
        ("Hello world", "world", True),  # "world" is in source_text
        ("Hello world", "planet", False),  # "planet" is not in source_text
        ("Привет мир", "мир", True),  # "мир" is in source_text
        ("Привет мир", "hello", False),  # "hello" is not in source_text
        ("Mixed case", "mixed", True),  # ignore case
        ("Numbers 123", "123", False),  # ignore numbers
        ("", "test", False),  # empty source_text
        ("test", "", False),  # empty target_text
    ],
)
def test_contains_foreign_words_basic(text_cleaner, source_text, target_text, expected):
    assert text_cleaner.contains_foreign_words(source_text, target_text) == expected


@pytest.mark.parametrize(
    "source_text, target_text, expected",
    [
        # Complex cases
        ("Hello world", "[IGNORE] world", True),  # Patterns are removed
        ("Hello world", "<div>world</div>", True),  # HTML tags are removed
        ("Hello world", "%VAR world", True),  # Variables are removed
        ("Hello world", "world %VAR", True),  # Variables are removed
        ("Hello world", "test123", False),  # Words with numbers are ignored
        ("Hello world", "hello-world", True),  # Dashes are taken into account
        ("Hello world", "hello_world", False),  # Underscores are ignored
    ],
)
def test_contains_foreign_words_complex(
    text_cleaner, source_text, target_text, expected
):
    assert text_cleaner.contains_foreign_words(source_text, target_text) == expected


@pytest.mark.parametrize(
    "source_text, target_text, expected",
    [
        # Edge cases
        ("", "", False),  # Both texts are empty
        ("   ", "test", False),  # Only spaces in source_text
        ("test", "   ", False),  # Only spaces in target_text
        ("Hello world", "hello, world!", True),  # Punctuation is removed
        ("Hello world", "WORLD", True),  # Case is ignored
        ("Hello world", "hello123", False),  # Numbers are ignored
    ],
)
def test_contains_foreign_words_edge_cases(
    text_cleaner, source_text, target_text, expected
):
    assert text_cleaner.contains_foreign_words(source_text, target_text) == expected


@pytest.mark.parametrize(
    "source_text, target_text, expected",
    [
        # Patterns in target_text
        ("Hello world", "[IGNORE] world", True),  # Removing square brackets
        ("Hello world", "<div>world</div>", True),  # Removing HTML tags
        ("Hello world", "%VAR world", True),  # Removing variables
        ("Hello world", "~action(world)", False),  # Removing with ~
        ("Hello world", "#~mission(world)", False),  # Removing with #~
    ],
)
def test_contains_foreign_words_pattern_removal(
    text_cleaner, source_text, target_text, expected
):
    assert text_cleaner.contains_foreign_words(source_text, target_text) == expected
