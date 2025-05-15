import pytest
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from src.config import ConfigManager
from src.preprocess.json_cleaner import JsonCleaner


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")


@pytest.fixture
def json_cleaner(tokenizer):
    return JsonCleaner(
        tokenizer=tokenizer,
        protected_patterns=ConfigManager().translation_config.protected_patterns,
        max_model_length=512,
        logger=MagicMock(),
    )


@pytest.mark.parametrize(
    "input_text, remove_patterns, expected",
    [
        # Базовые случаи без удаления шаблонов
        ("Welcome to ~mission(Location)", False, "Welcome to ~mission(Location)"),
        ("Press #~action(Button)", False, "Press #~action(Button)"),
        # Удаление шаблонов
        ("Welcome to ~mission(Location)", True, "Welcome to"),
        ("Press #~action(Button)", True, "Press"),
        # Обработка переносов строк
        ("Line1\\nLine2", False, "Line Line"),
        ("Line 3\\nLine 4", False, "Line 3 Line 4"),
        ("Double\\\\n", False, "Double"),
        # Очистка пробелов
        ("   Hello   ", False, "Hello"),
        ("Line  \n  next", False, "Line next"),
        ("Release \nRelease", False, "Release Release"),
        # HTML сущности
        ("<>&amp;", False, ""),
        ("test&nbsp;test", False, "test test"),
        # Символы и пунктуация
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
        # Граничные условия
        ("", False, ""),
        (None, False, ""),
        ("\n\n\n", False, ""),
        ("   ", False, ""),
    ],
)
def test_clean_text(json_cleaner, input_text, remove_patterns, expected):
    assert json_cleaner.clean_text(input_text, remove_patterns) == expected


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
def test_clean_text_complex(json_cleaner, input_text, remove_patterns, expected):
    assert (
        json_cleaner.clean_text(input_text, remove_patterns).strip() == expected.strip()
    )


@pytest.mark.parametrize(
    "source_text, target_text, expected",
    [
        # Базовые случаи
        ("Hello world", "world", True),  # "world" есть в source_text
        ("Hello world", "planet", False),  # "planet" нет в source_text
        ("Привет мир", "мир", True),  # "мир" есть в source_text
        ("Привет мир", "hello", False),  # "hello" нет в source_text
        ("Mixed case", "mixed", True),  # Регистр не важен
        ("Numbers 123", "123", False),  # Числа игнорируются
        ("", "test", False),  # Пустой source_text
        ("test", "", False),  # Пустой target_text
    ],
)
def test_contains_foreign_words_basic(json_cleaner, source_text, target_text, expected):
    assert json_cleaner.contains_foreign_words(source_text, target_text) == expected


@pytest.mark.parametrize(
    "source_text, target_text, expected",
    [
        # Сложные случаи
        ("Hello world", "[IGNORE] world", True),  # Шаблоны удаляются
        ("Hello world", "<div>world</div>", True),  # HTML теги удаляются
        ("Hello world", "%VAR world", True),  # Переменные удаляются
        ("Hello world", "world %VAR", True),  # Переменные удаляются
        ("Hello world", "test123", False),  # Слова с числами игнорируются
        ("Hello world", "hello-world", True),  # Дефисы учитываются
        ("Hello world", "hello_world", False),  # Подчеркивания игнорируются
    ],
)
def test_contains_foreign_words_complex(
    json_cleaner, source_text, target_text, expected
):
    assert json_cleaner.contains_foreign_words(source_text, target_text) == expected


@pytest.mark.parametrize(
    "source_text, target_text, expected",
    [
        # Граничные случаи
        ("", "", False),  # Оба текста пустые
        ("   ", "test", False),  # Только пробелы в source_text
        ("test", "   ", False),  # Только пробелы в target_text
        ("Hello world", "hello, world!", True),  # Знаки препинания удаляются
        ("Hello world", "WORLD", True),  # Регистр не важен
        ("Hello world", "hello123", False),  # Числа игнорируются
    ],
)
def test_contains_foreign_words_edge_cases(
    json_cleaner, source_text, target_text, expected
):
    assert json_cleaner.contains_foreign_words(source_text, target_text) == expected


@pytest.mark.parametrize(
    "source_text, target_text, expected",
    [
        # Шаблоны в target_text
        ("Hello world", "[IGNORE] world", True),  # Удаление квадратных скобок
        ("Hello world", "<div>world</div>", True),  # Удаление HTML тегов
        ("Hello world", "%VAR world", True),  # Удаление переменных
        ("Hello world", "~action(world)", False),  # Удаление с ~
        ("Hello world", "#~mission(world)", False),  # Удаление с #~
    ],
)
def test_contains_foreign_words_pattern_removal(
    json_cleaner, source_text, target_text, expected
):
    assert json_cleaner.contains_foreign_words(source_text, target_text) == expected
