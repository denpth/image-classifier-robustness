# Robust CIFAR-10 CNN Under Distribution Shift

## Project context

Developed during Western Governors University coursework. This repository is a record of my implementation and learning, not a client engagement, deployed operational system, or claim of a new algorithm. Supplied requirements, starter components, datasets, and third-party libraries are distinct from my own implementation. Where their exact provenance or redistribution terms have not been established, they are not represented as my original work or as cleared for public release.


A compact convolutional neural-network project that explores the trade-off between clean-image accuracy and robustness to corrupted inputs. The work compares a baseline CNN against a robustness-oriented variant trained with stronger in-model augmentation, then evaluates how both models behave on clean CIFAR-10 data and corruption stress tests.

The architecture is VGG-inspired but intentionally compact, using Global Average Pooling instead of a large fully connected head. The robust model incorporates augmentation such as noise, contrast, and brightness perturbations directly into the training graph.

## What this project demonstrates

- CNN design and training with TensorFlow/Keras
- Hyperparameter tuning with Keras Tuner
- Data-quality checks and preprocessing with NumPy/OpenCV
- Baseline-versus-robust training experiments
- Evaluation under image corruption and distribution shift
- Confusion-matrix, prediction, and training-curve analysis
- Exporting trained models as portable `.keras` artifacts

## Repository contents

- `D802.ipynb` — primary experiment notebook; the legacy filename is retained for reproducibility.
- `final_analysis.ipynb` — final analysis and comparison notebook.
- `CIFAR_load_and_clean.py` — downloads/prepares CIFAR-10 and writes the cleaned dataset used by the notebooks.
- `final_baseline.keras` — saved baseline model.
- `final_robust.keras` — saved robustness-oriented model.
- `corruptions_results.csv` — corruption-evaluation results.
- `final_predictions.csv` — saved prediction output.
- Historical tuner checkpoints remain in the private archive; final models and result CSVs are retained here.
- `confusion_matrix.png`, `confusion_matrixes.png`, and training-curve images — representative evaluation outputs.

The raw CIFAR datasets are intentionally **not committed to Git**. Generate the required cleaned dataset locally using the steps below.

## Environment setup

Python 3.10+ is recommended. Create an isolated environment and install the project dependencies:

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Download CIFAR-10

CIFAR-10 is distributed by the University of Toronto. The cleaner uses TensorFlow/Keras' official CIFAR-10 loader, so the simplest way to download the dataset into the location Keras expects is:

```bash
python -c "from tensorflow.keras.datasets import cifar10; cifar10.load_data(); print('CIFAR-10 downloaded and cached successfully')"
```

The source dataset is also available from the official CIFAR page:

https://www.cs.toronto.edu/~kriz/cifar.html

The explicit download step is optional. If CIFAR-10 is not already present in the local Keras cache, `CIFAR_load_and_clean.py` will download it automatically on first run.

## Build the cleaned dataset

Run the preprocessing pipeline from the repository root:

```bash
python CIFAR_load_and_clean.py
```

The cleaner performs the following pipeline:

1. Loads CIFAR-10 through `tensorflow.keras.datasets.cifar10`.
2. Removes invalid/NaN samples.
3. Applies the image-quality gate for artifacts, excessive noise, blur, and shape consistency.
4. Repairs localized artifact regions when appropriate.
5. Normalizes pixel values to `[0, 1]`.
6. One-hot encodes the class labels.
7. Creates a stratified training/validation split.
8. Writes the processed arrays to `cifar10_cleaned.npz`.

The generated `cifar10_cleaned.npz` file is intentionally ignored by Git because it is derived data and can be reproduced locally.

On a headless system or CI runner, use Matplotlib's non-interactive backend:

```bash
MPLBACKEND=Agg python CIFAR_load_and_clean.py
```

## Run the analysis

After `cifar10_cleaned.npz` exists, launch Jupyter:

```bash
jupyter lab
```

Open `D802.ipynb` and run the cells in order. `final_analysis.ipynb` contains the final comparison and reporting workflow.

GPU acceleration is strongly recommended for retraining and hyperparameter search. CPU-only execution is possible but substantially slower.

## CIFAR-10-C note

CIFAR-10-C is a separate corruption benchmark used to evaluate robustness under distribution shift. Its raw dataset is not stored in this repository. Existing corruption-result artifacts are retained so the completed experiment can be inspected without committing the large benchmark dataset.

## Model strategy

The project uses two related models:

**Baseline model** — trained on clean CIFAR-10 with conventional augmentation to establish the clean-data reference point.

**Robust model** — trained with a stronger stochastic augmentation curriculum intended to reduce sensitivity to corruptions and texture-specific shortcuts.

The comparison is meant to make the accuracy-versus-robustness trade-off visible rather than treating a single clean-set accuracy score as the only measure of model quality.

## Verification

GitHub Actions validates the Python cleaner, parses the committed notebooks as valid notebook JSON, runs the preprocessing functions on synthetic CIFAR-shaped data, and verifies that both saved Keras models can be loaded.
## Public portfolio release

This is a publication copy of my WGU coursework implementation and subsequent revisions. The original development history is retained privately. Assessment documents, student identifiers, and training-data snapshots are excluded from this public history. Notebook outputs are cleared; recorded evaluation artifacts are retained separately where available. The academic origin and third-party dependencies remain attributed.
