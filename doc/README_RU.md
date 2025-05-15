<div align="center">
  <h1>VerseBridge</h1>

  <p>Инструмент для перевода текстов Star Citizen на основе дообученной модели NLLB-200</p>
  
  <img src="https://github.com/user-attachments/assets/cde49eaa-f857-4be0-a2c7-215bd9c0a471" width="100">
</div>

<div align="center">
   <i>Это неофициальный фан-сайт Star Citizen, не аффилированный с Cloud Imperium Games. Весь контент, не созданный владельцем или пользователями, принадлежит их правообладателям.</i>
</div>

<br>

> **Проверено на**: _WSL Ubuntu 22.04_, _NVIDIA CUDA 12.8_, _12GB 4070 GPU_

**Документация**: [English](../README.md)

Это инструмент **перевода текстов** для игровых нужд, основанный на [NLLB-200 model](https://huggingface.co/facebook/nllb-200-distilled-1.3B) с дополнительным дообучением на датасете, который может быть создан на основе существующего перевода.

### Возможности

- **Модель NLLB-200**: Поддержка перевода на множество языков с высокой точностью.
- **Дообучение для Star Citizen**: Обучение на игровых переводах для контекстуально корректных результатов.
- **Мультиязычная поддержка**: Перевод на разные языки с использованием кодов FLORES-200.
- **Модульный пайплайн**: Отдельные скрипты для препроцессинга, обучения и перевода.
- **Интеграция LoRA**: Эффективное дообучение с Low-Rank Adaptation для экономии ресурсов.

## Установка

> Требуется Python 3.10 и NVIDIA GPU с поддержкой CUDA для оптимальной работы.

1. Установите UV:
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   В Windows:
   ```sh
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Клонируйте репозиторий:
   ```sh
   git clone https://github.com/sc-localization/VerseBridge.git
   cd VerseBridge
   ```
3. Установите зависимости:
   ```sh
   uv sync
   ```

## Использование

### Препроцессинг данных

Для подготовки данных к обучению или переводу запустите пайплайн препроцессинга из корня проекта:

```sh
uv run -m scripts.run_preprocess
```

**Шаги**:

1. **Настройте пути**:

   - Обновите `src/config/paths.py`, указав пути к исходному и целевому `.ini` файлам (например, `global_original.ini`, `global_pre_translated.ini`).
   - Установите `target_lang_code` в `src/config/language.py` (например, `rus_Cyrl` для русского) с использованием [FLORES-200 codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200).

2. **Разместите файлы**:

   - Скопируйте исходный (`global_original`) и переведённый (`global_pre_translated`) `.ini` файлы в папку `data/`.

3. **Запустите препроцессинг**:
   - **Конвертация `.ini` в JSON**:
     Результат: `data/data.json`.
   - **Очистка данных**:
     Результат: `data/cleaned_data.json`. Удаляет дубликаты, пустые строки и слишком длинные тексты.
   - **Разделение данных**:
     Результат: `data/train.json` (80%), `data/valid.json` (20%).

**Формат датасета**:

```json
{
  "source": "This is a test sentence",
  "target": "Это тестовое предложение"
}
```

### Обучение модели

Для дообучения NLLB-200 запустите:

```sh
uv run -m scripts.run_training
```

**Примечания:**

- Препроцессинг запускается автоматически, если отсутствуют файлы train/test JSON.
- Во время оценки считаются метрики (BLEU, ChrF, METEOR, BERTScore).
- Чекпоинты и логи сохраняются в указанных директориях.
- Включён ранний останов с patience=5 и threshold=0.001.

**Мониторинг обучения**:

```sh
uv run tensorboard --logdir logs/
```

**Конфигурация**:

- Изменяйте параметры обучения в `src/config/training.py` (например, эпохи, размер батча).
- Убедитесь, что `data/train.json` и `data/valid.json` готовы.
- Чекпоинты и результаты сохраняются в `models/`.

### Перевод файлов

Для перевода `.ini` файлов запустите:

```sh
uv run -m scripts.run_translation
```

**Примечания:**

- INI-файлы обрабатываются с сохранением защищённых паттернов (например, плейсхолдеры, переносы строк).
- Длинные тексты разбиваются с учётом лимита токенов модели.
- Логи сохраняются в указанной директории.
- Если не указан --input_path, скрипт обработает все INI-файлы в исходной директории.

**Результат**:

- Переведённые файлы сохраняются в `data/translated/<lang_code>/` (например, `data/translated/ru/global_original.ini`).

**Конфигурация**:

- Установите `target_lang_code` и `translate_dest_dir` в `src/config/translation.py`.

### Использование через CLI

**Примечание:** используйте атрибут `--help` для справки по аргументам

**1. Обучение модели (run_training.py)**

- обучение с LoRA:

```sh
uv run -m scripts.run_training --with-lora
```

- возобновить обучение с чекпоинта (если есть):

```sh
uv run -m scripts.run_training --with-lora --model-path models/lora/checkpoints/checkpoints-100
```

- обучение без LoRA на базовой модели:

```sh
uv run -m scripts.run_training
```

**2. Перевод (run_translation.py)**

**Примеры:**

- перевести все INI-файлы в исходной директории:

```sh
uv run -m scripts.run_translation --src-lang en --tgt-lang ru --translated_file_name translated.ini
```

- перевести INI-файл из пользовательской директории:

```sh
uv run -m scripts.run_translation --input-file-path data/global_original_test.ini
```

- использовать дообученную модель для перевода:

```sh
uv run -m scripts.run_translation --model-path models/base_model/result
```

- перевод с настройками по умолчанию:

```sh
uv run -m scripts.run_translation
```

## Вклад

Если вы хотите внести вклад в проект, ознакомьтесь с [CONTRIBUTING](../CONTRIBUTING.md).

## Лицензия

- **Код**: MIT License ([LICENSE_CODE](../LICENSE_CODE))
  <!-- TODO: Add if a training dataset created from translations will be added to the repository -->
  <!-- - **Ru Translations**: Creative Commons BY-NC-SA 4.0 ([LICENSE_TRANSLATIONS](../LICENSE_TRANSLATIONS)) -->

**Cloud Imperium Games предоставлено специальное разрешение на неограниченное использование. См. [SPECIAL_PERMISSION](../SPECIAL_PERMISSION.md).**
