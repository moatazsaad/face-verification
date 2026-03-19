# Face Verification Project

## Overview
This project implements a face verification system using the Labeled Faces in the Wild (LFW) dataset. It includes:

- Deterministic image ingestion and splitting into train/validation/test sets.
- Automatic generation of positive and negative image pairs for verification.
- Calculation of similarities using Cosine Similarity and Euclidean Distance.
- Performance benchmarking: Python loop-based implementation vs. vectorized NumPy implementation.
- Reproducible outputs with saved pair splits for train, validation, and test sets.

## Project Structure

```

face-verification/
│
├─ src/
│ ├─ data_ingest.py        # Load and split LFW dataset
│ ├─ pair_gen.py           # Generate and save positive/negative image pairs
│ ├─ similarity.py         # Similarity functions (Python loop & NumPy)
│ ├─ benchmark.py          # Benchmark comparing Python loops vs NumPy
│ └─ config.py             # Config file for seed, ratios and output directory
│
├─ scripts/
│ └─ run_pipeline.py       # Runs the full pipeline: ingestion, pair generation, benchmarking
│
├─ notebooks/
│ └─ notebook.ipynb
│
├─ artifacts/              # Output directory for dataset manifest, pairs and labels
├─ pyproject.toml          # Project metadata and dependencies
└─ README.md               # This file

````

## How to Run

### 1. Install required packages
Using **uv** (recommended):

```bash
uv sync
````

Or using **pip** (works on Windows/Mac/Linux):

```bash
pip install tensorflow-datasets numpy
```

### 2. Run the full pipeline (recommended)

```bash
uv run -m scripts.run_pipeline
```

This will:

* Generate deterministic LFW splits.
* Generate and save positive/negative pairs for train, validation, and test sets.
* Run similarity benchmarks (Python loop vs NumPy).
* Print runtime comparisons and correctness verification.
### 3. Optional: Run steps individually

* **Data ingestion**:

```bash
uv run -m src.data_ingest
```

* **Generate image pairs**:

```bash
uv run -m src.pair_gen

```

* **Run similarity benchmark**:

```bash
uv run -m src.benchmark
```

> These individual steps are only necessary if you want to test or debug specific parts. For Milestone 1 submission, running the full pipeline is sufficient.

### Mac Instructions

All `uv` commands work the same on Mac. Ensure you activate your virtual environment first:

```bash
source .venv/bin/activate
uv sync
uv run -m scripts.run_pipeline
```

## Output

Saved automatically in `artifacts/`:

* `dataset_manifest.json` – Dataset info and split sizes.
* `train_pairs.npy`, `train_labels.npy` – Training pairs and labels.
* `val_pairs.npy`, `val_labels.npy` – Validation pairs and labels.
* `test_pairs.npy`, `test_labels.npy` – Test pairs and labels.

Benchmark prints:

* Time taken for Python loops and NumPy vectorized operations.
* Speedup factor of NumPy vs loops.
* Correctness check confirmation for both similarity measures.

```


milestone 2:
The operating threshold is selected by maximizing balanced accuracy on the validation split.
Update the README (very important)

Explain briefly:

Dataset

LFW dataset

validation negative sampling

3:1 negative:positive ratio

deterministic seed

Threshold selection

cosine similarity

threshold sweep

selection rule:

maximize balanced_accuracy
Evaluation pipeline
validation sampled set → threshold sweep → best threshold
test set → final evaluation
4. Document the runs

In README add a small table:

run_id	purpose
baseline_val_eval	baseline evaluation
val_threshold_sweep	initial sweep
sampled_val_sweep	sweep after negative sampling
sampled_test_eval	final test evaluation

Confirm pipeline explanation

Your README should show:

baseline_val_sweep
        ↓
sample_validation_pairs
        ↓
sampled_val_sweep
        ↓
val_sampled_best_threshold
        ↓
sampled_test_eval