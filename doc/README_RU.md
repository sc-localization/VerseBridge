<div align="center">
	<h1>VerseBridge</h1>

  <p>Инструмент для автоматизации перевода и анализа игровых текстов с использованием современных моделей машинного обучения. Поддерживает многоязычный перевод, распознавание именованных сущностей (NER), а также обучение и адаптацию моделей для игровых задач и пользовательских датасетов.</p>

  <img src="https://github.com/user-attachments/assets/cde49eaa-f857-4be0-a2c7-215bd9c0a471" width="100">
</div>

<div align="center">
	<i>Неофициальный фан-сайт Star Citizen, не аффилирован с Cloud Imperium Games. Весь контент, не созданный владельцем или пользователями сайта, принадлежит их правообладателям.</i>
</div>

<br>

> **Проверено на**: _WSL Ubuntu 22.04_, _NVIDIA CUDA 12.8_, _12GB 4070 GPU_

**Документация**: [English](../README.md)

---

**Модуль перевода**

Автоматизирует перевод игровых текстов между языками с помощью дообученных моделей машинного обучения. Используется для локализации ресурсов, поддерживает сохранение структуры и специальных конструкций в исходных файлах. Обеспечивает быстрый и качественный машинный перевод с учетом игрового контекста.

**Модуль NER (распознавание именованных сущностей)**

Позволяет извлекать именованные сущности (например, имена, организации, игровые объекты) из текстов. Это важно для автоматической разметки, анализа, а также для повышения качества перевода — например, чтобы не переводить имена собственные или использовать их для дополнительной адаптации модели.

**Связь модулей**

NER может использоваться как вспомогательный этап перед переводом: сначала из текста извлекаются сущности, которые затем можно защитить от перевода или обработать отдельно. Это помогает избежать ошибок при переводе имен, терминов и других важных элементов, а также повышает итоговое качество локализации.

## Установка

> Требуется Python 3.10 и NVIDIA GPU с поддержкой **CUDA 12.8**.

**Примечание:** CUDA 12.8 необходим для совместимости с pytorch

Для установки CUDA на WSL Ubuntu 22.04 следуйте [инструкции](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network) или выполните:

```sh
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt update
  sudo apt -y install cuda-toolkit-12-8
```

1. Установите UV:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   или в Windows:

   ```sh
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Клонируйте репозиторий:

   ```sh
   git clone https://github.com/sc-localization/VerseBridge.git
   cd VerseBridge
   ```

3. Установите необходимые зависимости:

   ```sh
   uv sync
   ```

## Использование

---

### Процесс перевода

#### 1. Предобработка данных

Для подготовки данных для обучения или перевода запустите pipeline предобработки из корня проекта:

```sh
uv run -m scripts.run_preprocess
```

**Этапы:**

1. **Настройка путей**:

- Обновите `src/config/paths.py` с путями к оригинальным и переведённым `.ini` файлам (например, `global_original.ini`, `global_pre_translated.ini`).
- Установите `target_lang_code` в `src/config/language.py` (например, `rus_Cyrl` для русского) используя [FLORES-200 codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200).

2. **Размещение файлов**:

- Скопируйте оригинальный (`global_original`) в `data/raw/en` и переведённый (`global_pre_translated`) `.ini` в `data/raw/{target_lang_code}`.

3. **Запуск предобработки**:

- **Конвертация `.ini` в JSON**: Выход: `data/raw/training/{source_lang_code-target_lang_code}/data.json`.
- **Очистка данных**: Выход: `data/data/raw/training/{source_lang_code-target_lang_code}/cleaned_data.json`. Удаляет дубликаты, пустые строки и слишком длинные тексты.
- **Разделение данных**: Выход: `data/data/raw/training/{source_lang_code-target_lang_code}/train.json` (80%), `data/data/raw/training/{source_lang_code-target_lang_code}/valid.json` (20%).

**Формат датасета:**

```json
{
  "original": "This is a test sentence",
  "translated": "Это тестовое предложение"
}
```

#### 2. Обучение модели

Для дообучения модели NLLB-200 выполните:

```sh
uv run -m scripts.run_training
```

**Примечания:**

- Предобработка запускается автоматически, если отсутствуют файлы для обучения/тестирования.
- Метрики (BLEU, ChrF, METEOR, BERTScore) вычисляются во время оценки.
- Чекпоинты и логи сохраняются в указанных директориях.
- Включён early stopping с patience=5 и threshold=0.001.

**Мониторинг обучения:**

```sh
uv run tensorboard --logdir logs/
```

**Конфигурация:**

- Обновите `src/config/training.py` для параметров обучения (например, эпохи, размер батча).
- Убедитесь, что `data/train.json` и `data/valid.json` готовы.
- Чекпоинты и результаты сохраняются в `models/`.

#### 3. Перевод файлов

Для перевода `.ini` файлов выполните:

```sh
uv run -m scripts.run_translation
```

**Примечания:**

- INI-файлы обрабатываются с сохранением защищённых паттернов (например, плейсхолдеры, переносы строк).
- Длинные тексты разбиваются с учётом лимитов токенов модели.
- Логи сохраняются в указанной директории.
- Если не указан --input_path, скрипт обработает все INI-файлы в исходной директории.

**Выходные данные:**

- Переведённые файлы сохраняются в `data/translated/<lang_code>/` (например, `data/translated/ru/global_original.ini`).

**Конфигурация:**

- Установите `target_lang_code` и `translate_dest_dir` в `src/config/translation.py`.

---

### Процесс NER (распознавание именованных сущностей)

#### 1. Извлечение и подготовка данных

Запустите предобработку и извлечение сущностей из исходных текстов:

```sh
uv run -m scripts.run_ner --stage preprocess
```

Будут созданы файлы для разметки и обучения в папке `data/ner/`:

- `ner_unannotated.json` — неразмеченные данные для ручной разметки
- `dataset_bio.json` — данные в формате BIO для обучения

#### 2. Ручная разметка и ревью

Для ручной проверки и корректировки разметки используйте веб-интерфейс (Streamlit):

```sh
uv run -m scripts.run_ner --stage review
```

После проверки будет создан файл `dataset_corrected.json`.

#### 3. Обучение модели NER

Для обучения модели на размеченных данных выполните:

```sh
uv run -m scripts.run_ner --stage train
```

Модель будет обучаться на файлах `train.json` и `test.json` (создаются автоматически из размеченного датасета).

#### 4. Извлечение сущностей из новых текстов

Для применения обученной модели к новым данным:

```sh
uv run -m scripts.run_ner --stage extract
```

**Формат данных**
Входные и выходные файлы для NER находятся в `data/ner/`:

- `dataset_bio.json` — данные в формате BIO для обучения
- `test.json`, `train.json` — разделённые датасеты
- `ner_unannotated.json` — неразмеченные данные для разметки
  Пример структуры данных:

```json
{
  "tokens": ["This", "is", "VerseBridge"],
  "labels": ["O", "O", "B-ORG"]
}
```

**Конфигурация**

- Основные параметры и пути задаются в `src/config/ner.py` и `src/config/paths.py`.
- Веб-интерфейс для ревью разметки использует Streamlit.

**Примечания**

- Для корректной работы требуется предобработка (`--stage preprocess`).
- Все логи сохраняются в папке `logs/`.

---

## Примеры CLI

**Примечание:** используйте атрибут `--help` для получения справки по доступным аргументам

### CLI перевода

- обучение с LoRA:

```sh
uv run -m scripts.run_training --with-lora
```

- продолжить обучение с чекпоинта (если чекпоинт существует):

```sh
uv run -m scripts.run_training --with-lora --model-path models/lora/checkpoints/checkpoints-100
```

- обучение без LoRA на базовой модели

```sh
uv run -m scripts.run_training
```

- перевод всех INI-файлов в исходной директории:

```sh
uv run -m scripts.run_translation --src-lang en --tgt-lang ru --translated-file-name translated.ini
```

- перевод INI-файла из пользовательской директории:

```sh
uv run -m scripts.run_translation --input-file data/raw/global_original_test.ini
```

- повторный перевод уже переведённого INI-файла:

```sh
uv run -m scripts.run_translation --input-file data/raw/global_original_test.ini --existing-translated-file data/global_original_exist.ini
```

- повторный перевод с приоритетом (если переведённые файлы уже есть в директории назначения `translation_results`, они будут заменены строками из `existing-translated-file`):

```sh
uv run -m scripts.run_translation --input-file data/raw/global_original_test.ini --existing-translated-file data/global_original_exist.ini
```

- использовать дообученную модель для перевода:

```sh
uv run -m scripts.run_translation --model-path models/base_model/result
```

- перевод с настройками по умолчанию:

```sh
uv run -m scripts.run_translation
```

### CLI NER

- Предобработка и извлечение сущностей:

```sh
uv run -m scripts.run_ner --stage preprocess
```

- Ручная разметка и ревью (Streamlit):

```sh
uv run -m scripts.run_ner --stage review
```

- Обучение модели NER:

```sh
uv run -m scripts.run_ner --stage train
```

- Извлечение сущностей из новых текстов:

```sh
uv run -m scripts.run_ner --stage extract
```

## Вклад

Если вы хотите внести вклад в проект, пожалуйста, прочитайте [CONTRIBUTING](CONTRIBUTING.md).

## Лицензия

- **Код**: MIT License ([LICENSE_CODE](../LICENSE))
  <!-- TODO: Добавить, если датасет из переводов будет добавлен в репозиторий -->
  <!-- - **Ru Translations**: Creative Commons BY-NC-SA 4.0 ([LICENSE_TRANSLATIONS](LICENSE_TRANSLATIONS)) -->

**Cloud Imperium Games предоставлено специальное разрешение на неограниченное использование. См. [SPECIAL_PERMISSION](../SPECIAL_PERMISSION.md).**
