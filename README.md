<div align="center">
  <h1>VerseBridge</h1>

  <p>A tool for automating translation and analysis of game texts using modern machine learning models. Supports multilingual translation, named entity recognition (NER), as well as training and adaptation of models for game tasks and custom datasets.</p>
  
  <img src="https://github.com/user-attachments/assets/cde49eaa-f857-4be0-a2c7-215bd9c0a471" width="100">
</div>

<div align="center">
  <i>This is an unofficial Star Citizen fansite, not affiliated with Cloud Imperium Games. All content not authored by the host or users belongs to its respective owners.</i>
</div>

<br>

> **Tested on**: _WSL Ubuntu 22.04_, _NVIDIA CUDA 12.8_, _12GB 4070 GPU_

**Documentation**: [Русский](doc/README_RU.md)

---

**Translation module**

Automates the translation of game texts between languages using fine-tuned machine learning models. Used for localization of resources, supports preservation of structure and special constructs in source files. Enables fast, high-quality machine translation with game context awareness.

**NER module (Named Entity Recognition)**

Allows extracting named entities (e.g., names, organizations, game objects) from texts. This is important for automatic annotation, analysis, and also for improving translation quality — for example, to avoid translating proper names or to use them for additional model adaptation.

**How modules are connected**

NER can be used as an auxiliary step before translation: first, entities are extracted from the text, which can then be protected from translation or processed separately. This helps avoid errors when translating names, terms, and other important elements, and improves the final localization quality.

## Installation

> Requires Python 3.10 and NVIDIA GPU with **CUDA 12.8** support.

**Note:** CUDA 12.8 is required for pytorch compatibility

To install CUDA on WSL Ubuntu 22.04 follow the [instructions](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network) or run:

```sh
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt update
  sudo apt -y install cuda-toolkit-12-8
```

1. Install UV:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   or in Windows:

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

---

### Translation Pipeline

#### 1. Preprocess Data

To prepare data for training or translation, run the preprocessing pipeline from the project root:

```sh
uv run -m scripts.run_preprocess
```

**Steps:**

1. **Configure Paths**:

- Update `src/config/paths.py` with paths to original and translated `.ini` files (e.g., `global_original.ini`, `global_pre_translated.ini`).
- Set `target_lang_code` in `src/config/language.py` (e.g., `rus_Cyrl` for Russian) using [FLORES-200 codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200).

2. **Place Files**:

- Copy original (`global_original`) to `data/raw/en` and translated (`global_pre_translated`) `.ini` to `data/raw/{target_lang_code}`.

3. **Run Preprocessing**:

- **Convert `.ini` to JSON**: Output: `data/raw/training/{source_lang_code-target_lang_code}/data.json`.
- **Clean Data**: Output: `data/data/raw/training/{source_lang_code-target_lang_code}/cleaned_data.json`. Removes duplicates, empty rows, and oversized tokenized text.
- **Split Data**: Output: `data/data/raw/training/{source_lang_code-target_lang_code}/train.json` (80%), `data/data/raw/training/{source_lang_code-target_lang_code}/valid.json` (20%).

**Dataset Format:**

```json
{
  "original": "This is a test sentence",
  "translated": "Это тестовое предложение"
}
```

#### 2. Train Model

To fine-tune the model, run:

```sh
uv run -m scripts.run_training
```

**Notes:**

- Preprocessing runs automatically if training/test JSON files are missing.
- Metrics (BLEU, ChrF, METEOR, BERTScore) are computed during evaluation.
- Checkpoints and logs are saved in the configured directories.
- Early stopping is enabled with a patience of 5 and threshold of 0.001.

**Monitor Training:**

```sh
uv run tensorboard --logdir logs/
```

**Configuration:**

- Update `src/config/training.py` for training parameters (e.g., epochs, batch size).
- Ensure `data/train.json` and `data/valid.json` are ready.
- Model checkpoints and results are saved in `models/`.

#### 3. Translate Files

To translate `.ini` files, run:

```sh
uv run -m scripts.run_translation
```

**Notes:**

- INI files are processed with protected patterns preserved (e.g., placeholders, newlines).
- Long texts are split to respect model token limits.
- Logs are saved in the configured logging directory.
- If no --input_path is provided, the script processes all INI files in the source directory.

**Output:**

- Translated files saved in `data/translated/<lang_code>/` (e.g., `data/translated/ru/global_original.ini`).

**Configuration:**

- Set `target_lang_code` and `translate_dest_dir` in `src/config/translation.py`.

---

### Named Entity Recognition (NER) Pipeline

#### 1. Data extraction and preparation

Run preprocessing and entity extraction from the source texts:

```sh
uv run -m scripts.run_ner --stage preprocess
```

This will create files for annotation and training in the `data/ner/` folder:

- `ner_unannotated.json` — unannotated data for manual labeling
- `dataset_bio.json` — data in BIO format for training

#### 2. Manual annotation and review

For manual review and correction of annotations, use the web interface (Streamlit):

```sh
uv run -m scripts.run_ner --stage review
```

After review, the file `dataset_corrected.json` will be created.

#### 3. Training the NER model

To train the model on annotated data, run:

```sh
uv run -m scripts.run_ner --stage train
```

The model will be trained on `train.json` and `test.json` files (created automatically from the annotated dataset).

#### 4. Entity extraction from new texts

To apply the trained model to new data:

```sh
uv run -m scripts.run_ner --stage extract
```

**Data format**
Input and output files for NER are located in `data/ner/`:

- `dataset_bio.json` — data in BIO format for training
- `test.json`, `train.json` — split datasets
- `ner_unannotated.json` — unannotated data for annotation
  Example data structure:

```json
{
  "tokens": ["This", "is", "VerseBridge"],
  "labels": ["O", "O", "B-ORG"]
}
```

**Configuration**

- Main parameters and paths are set in `src/config/ner.py` and `src/config/paths.py`.
- The web interface for annotation review uses Streamlit.

**Notes**

- Preprocessing is required for correct operation (`--stage preprocess`).
- All logs are saved in the `logs/` folder.

---

## CLI Usage Examples

**Note:** use the `--help` attribute to get help about available arguments

### Translation CLI

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

- translate all INI files in source directory:

```sh
uv run -m scripts.run_translation --src-lang en --tgt-lang ru --translated-file-name translated.ini
```

- translate INI file from custom directory:

```sh
uv run -m scripts.run_translation --input-file data/raw/global_original_test.ini
```

- use a fine tuned model for translation:

```sh
uv run -m scripts.run_translation --model-path models/base_model/result
```

- translation with default settings:

```sh
uv run -m scripts.run_translation
```

### NER CLI

- Preprocess and extract entities:

```sh
uv run -m scripts.run_ner --stage preprocess
```

- Manual annotation and review (Streamlit):

```sh
uv run -m scripts.run_ner --stage review
```

- Train NER model:

```sh
uv run -m scripts.run_ner --stage train
```

- Extract entities from new texts:

```sh
uv run -m scripts.run_ner --stage extract
```

## Contribution

If you would like to contribute to the project, please read [CONTRIBUTING](CONTRIBUTING.md).

## License

- **Code**: MIT License ([LICENSE_CODE](LICENSE))
  <!-- TODO: Add if a training dataset created from translations will be added to the repository -->
  <!-- - **Ru Translations**: Creative Commons BY-NC-SA 4.0 ([LICENSE_TRANSLATIONS](LICENSE_TRANSLATIONS)) -->

**Special permission is granted to Cloud Imperium Games for unrestricted use. See [SPECIAL_PERMISSION](SPECIAL_PERMISSION.md).**
