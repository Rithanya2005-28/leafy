# LeafLens — Plant Leaf Classification System

A Flask web application that identifies leaf species from uploaded photos using a
MobileNetV2 transfer-learning model, OpenCV preprocessing, a small compiler-style
command parser, and SHA-256 upload integrity checks.

The model is trained on the **Kaggle Leaf Classification** dataset: **990 labelled
images across 99 tree species** (10 images per species). A stratified
train/validation/test split is created by `prepare_dataset.py`.

## Features

- Drag-and-drop image upload with preview, validation (JPG/JPEG/PNG, max 5 MB)
  and reset.
- Real predictions with confidence score and top-3 alternatives.
- Low-confidence warning when the best match is below 50 %.
- Processed image preview (224 × 224, RGB, Gaussian blur, normalized).
- SHA-256 fingerprint of every upload plus integrity verification.
- Compiler-style command mode: `CLASSIFY <filename>` with lexical, syntax and
  semantic analysis.
- REST API: `/classify`, `/command`, `/api/info`, `/api/classes`, `/uploads/<file>`.
- Real evaluation metrics and a 99 × 99 confusion matrix produced after training.

## Project structure

```text
Plant-Leaf-Classification/
│   app.py                    # Flask app + REST API
│   prepare_dataset.py        # Builds dataset/ and dataset_split/ from Kaggle files
│   train_model.py            # Transfer learning, evaluation, metrics + confusion matrix
│   model_loader.py           # Lazy TensorFlow model loader
│   test_modules.py           # Unit tests
│   run_server.bat            # Windows helper to start the server
│   requirements.txt
│
├── model/
│   ├── leaf_classifier.keras     # trained model (generated)
│   ├── class_names.json          # ordered class labels (generated)
│   ├── metrics.json              # test accuracy / per-class report (generated)
│   ├── confusion_matrix.png      # generated
│   └── confusion_matrix.csv      # generated
│
├── dataset/                  # full labelled dataset, one folder per species
├── dataset_split/
│   ├── train/                # 693 images (7 per class)
│   ├── validation/           # 198 images (2 per class)
│   └── test/                 #  99 images (1 per class)
│
├── images/images/            # raw Kaggle images (1,584 files)
├── train.csv/train.csv       # id → species labels (990 rows)
├── test.csv/test.csv         # unlabelled images (594 rows)
├── legacy_dataset/           # original 5-class sample images (preserved)
│
├── compiler/command_parser.py
├── image_processing/preprocessing.py
├── security/image_security.py
├── templates/index.html
├── static/css/style.css
├── static/js/script.js
└── uploads/                  # uploaded images (created at runtime)
```

## Requirements

- **Python 3.12** — TensorFlow 2.21 does not publish wheels for Python 3.14.
- On Windows, install the virtual environment at a **short path**. This project
  lives under OneDrive and TensorFlow's file names exceed the 260-character
  Windows path limit when installed inside the project folder.

## Installation

```powershell
cd "C:\Users\<you>\OneDrive\Desktop\leaf classification\Plant-Leaf-Classification"

py -3.12 -m venv "$env:USERPROFILE\leaf-venv"
& "$env:USERPROFILE\leaf-venv\Scripts\python.exe" -m pip install --upgrade pip
& "$env:USERPROFILE\leaf-venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Prepare the dataset

```powershell
& "$env:USERPROFILE\leaf-venv\Scripts\python.exe" prepare_dataset.py
```

This reads `train.csv/train.csv` (or `train.csv`), copies the 990 labelled images
into `dataset/<species>/`, creates the stratified `dataset_split/` folders, saves
`model/class_names.json`, and moves non-Kaggle sample folders to `legacy_dataset/`
instead of deleting them. Re-running is safe.

## Train and evaluate

```powershell
& "$env:USERPROFILE\leaf-venv\Scripts\python.exe" train_model.py
```

The script trains a frozen MobileNetV2 head for up to 12 epochs, fine-tunes the
top 30 layers for 4 epochs, restores the best validation checkpoint, and then
evaluates on the 99-image test split. Outputs:

- `model/leaf_classifier.keras`
- `model/metrics.json` — test accuracy, top-3 accuracy, macro F1, per-class report
- `model/confusion_matrix.png` / `.csv`

## Run the server

```powershell
& "$env:USERPROFILE\leaf-venv\Scripts\python.exe" app.py
```

or double-click `run_server.bat`. Open <http://127.0.0.1:5000>.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/api/info` | Model status, class count, test accuracy |
| GET | `/api/classes` | All 99 species labels |
| POST | `/classify` | Multipart form field `file` → prediction JSON |
| POST | `/command` | JSON `{"command": "CLASSIFY 123.jpg"}` |
| GET | `/uploads/<filename>` | Previously uploaded image |
| GET | `/health` | Simple health check |

Example (Windows PowerShell — use `curl.exe`, not the `curl` alias):

```powershell
curl.exe -X POST -F "file=@dataset_split/test/Acer_Opalus/1.jpg" http://127.0.0.1:5000/classify
```

```powershell
curl.exe -X POST -H "Content-Type: application/json" -d "{\"command\":\"CLASSIFY 1.jpg\"}" http://127.0.0.1:5000/command
```

## Tests

```powershell
& "$env:USERPROFILE\leaf-venv\Scripts\python.exe" -m unittest -v test_modules.py
```

Tests cover tokenization/parsing errors, path-traversal rejection, SHA-256
verification, preprocessing shape/range, and project structure.

## Notes and limitations

- Only 10 labelled images exist per species, so accuracy is limited; adding more
  photos per class is the most effective improvement.
- `test.csv` images are unlabelled — they are useful for manual demos but not for
  accuracy measurements.
- The 5 original sample classes (Mango, Neem, Guava, Tulsi, Jackfruit) are kept
  in `legacy_dataset/`.
- The development server is for local demos only; use a WSGI server for
  production.
