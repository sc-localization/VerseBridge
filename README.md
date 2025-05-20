<div align="center">
  <h1>VerseBridge</h1>

  <p>A text translation tool for Star Citizen, powered by the fine-tuned NLLB-200 model</p>
  
  <img src="https://github.com/user-attachments/assets/cde49eaa-f857-4be0-a2c7-215bd9c0a471" width="100">
</div>

<div align="center">
   <i>This is an unofficial Star Citizen fansite, not affiliated with Cloud Imperium Games. All content not authored by the host or users belongs to its respective owners</i>
</div>

<br>

> **Tested on**: _WSL Ubuntu 22.04_, _NVIDIA CUDA 12.8_, _12GB 4070 GPU_

**Documentation**: [Русский](doc/README_RU.md)

Is a **text translation** tool for game use, based on the [NLLB-200 model](https://huggingface.co/facebook/nllb-200-distilled-1.3B) with additional fine-tuning on a dataset that can be created from an existing translation.

### Features

- **NLLB-200 Model**: Supports translation across multiple languages with high accuracy.
- **Fine-Tuned for Star Citizen**: Trained on game-specific translations for context-aware results.
- **Multi-Language Support**: Translates into various languages using FLORES-200 codes.
- **Modular Pipeline**: Separate preprocessing, training, and translation scripts for flexibility.
- **LoRA Integration**: Efficient fine-tuning with Low-Rank Adaptation for resource optimization.

## Installation

> Requires Python 3.10 and NVIDIA GPU with CUDA support for optimal performance.

1. Inslall UV:
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   in Windows:
   ```sh
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Clone the repository:
   ```sh
   git clone https://github.com/sc-localization/VerseBridge.git
   cd VerseBridge
   ```
3. Install the required dependencies:
   ```sh
   uv sync
   ```

## Usage

### Preprocess Data

To prepare data for training or translation, run the preprocessing pipeline from the project root:

```sh
uv run -m scripts.run_preprocess
```

**Steps**:

1. **Configure Paths**:

   - Update `src/config/paths.py` with paths to original and translated `.ini` files (e.g., `global_original.ini`, `global_pre_translated.ini`).
   - Set `target_lang_code` in `src/config/language.py` (e.g., `rus_Cyrl` for Russian) using [FLORES-200 codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200).

2. **Place Files**:

   - Copy original (`global_original`) and translated (`global_pre_translated`) `.ini` files to `data/`.

3. **Run Preprocessing**:
   - **Convert `.ini` to JSON**:
     Output: `data/data.json`.
   - **Clean Data**:
     Output: `data/cleaned_data.json`. Removes duplicates, empty rows, and oversized tokenized text.
   - **Split Data**:
     Output: `data/train.json` (80%), `data/valid.json` (20%).

**Dataset Format**:

```json
{
  "original": "This is a test sentence",
  "translated": "Это тестовое предложение"
}
```

### Train Model

To fine-tune the NLLB-200 model, run:

```sh
uv run -m scripts.run_training
```

**Notes:**

- Preprocessing runs automatically if training/test JSON files are missing.
- Metrics (BLEU, ChrF, METEOR, BERTScore) are computed during evaluation.
- Checkpoints and logs are saved in the configured directories.
- Early stopping is enabled with a patience of 5 and threshold of 0.001.

**Monitor Training**:

```sh
uv run tensorboard --logdir logs/
```

**Configuration**:

- Update `src/config/training.py` for training parameters (e.g., epochs, batch size).
- Ensure `data/train.json` and `data/valid.json` are ready.
- Model checkpoints and results are saved in `models/`.

### Translate Files

To translate `.ini` files, run:

```sh
uv run -m scripts.run_translation
```

**Notes:**

- INI files are processed with protected patterns preserved (e.g., placeholders, newlines).
- Long texts are split to respect model token limits.
- Logs are saved in the configured logging directory.
- If no --input_path is provided, the script processes all INI files in the source directory.

**Output**:

- Translated files saved in `data/translated/<lang_code>/` (e.g., `data/translated/ru/global_original.ini`).

**Configuration**:

- Set `target_lang_code` and `translate_dest_dir` in `src/config/translation.py`.

### CLI Usage

**Note:** use the `--help` attribute to get help about available arguments

**1. Training a Model (run_training.py)**

- train with LoRA:

```sh
uv run -m scripts.run_training --with-lora
```

- resume training from a checkpoint (if checkpoint exist):

```sh
uv run -m scripts.run_training --with-lora --model-path models/lora/checkpoints/checkpoints-100
```

- train without LoRA using a base model

```sh
uv run -m scripts.run_training
```

**2. Performing Translation (run_translation.py)**

**Examples:**

- translate all INI files in source directory:

```sh
uv run -m scripts.run_translation --src-lang en --tgt-lang ru --translated_file_name translated.ini
```

- translate INI file from custom directory:

```sh
uv run -m scripts.run_translation --input-file-path data/global_original_test.ini
```

- use a fine tuned model for translation:

```sh
uv run -m scripts.run_translation --model-path models/base_model/result
```

- translation with default settings:

```sh
uv run -m scripts.run_translation
```

## Contribution

If you would like to contribute to the project, please read [CONTRIBUTING](CONTRIBUTING.md).

## License

- **Code**: MIT License ([LICENSE_CODE](LICENSE_CODE))
  <!-- TODO: Add if a training dataset created from translations will be added to the repository -->
  <!-- - **Ru Translations**: Creative Commons BY-NC-SA 4.0 ([LICENSE_TRANSLATIONS](LICENSE_TRANSLATIONS)) -->

**Special permission is granted to Cloud Imperium Games for unrestricted use. See [SPECIAL_PERMISSION](SPECIAL_PERMISSION.md).**
