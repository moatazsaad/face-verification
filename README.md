```markdown
# Face Verification Project

<<<<<<< HEAD
=======
**Group:** ML Verifier  
**Group Members:** Moataz Osama Saadeldin Abdelaziz, Ankan Roy  

---

>>>>>>> 20a6f7f (Update README)
## Overview

<<<<<<< HEAD
- Deterministic image ingestion and splitting into train/validation/test sets.
- Automatic generation of positive and negative image pairs for verification.
- Calculation of similarities using Cosine Similarity and Euclidean Distance.
- Performance benchmarking: Python loop-based implementation vs. vectorized NumPy implementation.
- Reproducible outputs with saved pair splits for train, validation, and test sets.
=======
This project implements a face verification system using the Labeled Faces in the Wild (LFW) dataset.

- **Milestone 1:** deterministic pipeline (data ingestion, pair generation, similarity scoring, benchmarking)  
- **Milestone 2:** reproducible evaluation system with threshold calibration, experiment tracking, data-centric iteration, and error analysis  

The system takes two face images and outputs:
- a similarity score  
- a same-person vs different-person decision based on a threshold  

---

## Milestone 1 (Baseline)

- Deterministic LFW ingestion and dataset split  
- Deterministic pair generation (train/validation/test)  
- Similarity computation (cosine similarity and Euclidean distance)  
- Benchmarking Python loops vs NumPy vectorization  

---

## Milestone 2 (Evaluation & Iteration)

### Baseline
- Threshold selected from full validation set  
- Rule: **maximize balanced accuracy**  
- Final evaluation on held-out test set  

### Data-Centric Improvement
- Modified validation distribution:
  - keep all positive pairs  
  - sample negatives to **3:1 ratio (negative:positive)**  
- Threshold re-selected on sampled validation  
- Test set unchanged  

### Key Result
Both baseline and improved systems produced the same threshold (**0.76**) and identical test results.

This indicates:
- similarity score ranking is stable  
- threshold selection is robust to validation sampling  

---
>>>>>>> 20a6f7f (Update README)

## Project Structure

```

face-verification/
│
├─ src/
<<<<<<< HEAD
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
=======
│ ├─ data_ingest.py
│ ├─ pair_gen.py
│ ├─ similarity.py
│ ├─ benchmark.py
│ ├─ evaluation.py
│ ├─ metrics.py
│ ├─ validation.py
│ ├─ run_tracker.py
│ └─ config.py
│
├─ scripts/
│ ├─ run_pipeline.py
│ ├─ run_data_ingest.py
│ ├─ run_pair_gen.py
│ ├─ run_benchmark.py
│ ├─ run_val_sweep.py
│ ├─ run_val_eval.py
│ ├─ run_baseline_test_eval.py
│ ├─ sample_validation_pairs.py
│ ├─ run_sampled_val_sweep.py
│ ├─ run_sampled_val_eval.py
│ └─ run_sampled_test_eval.py
│
├─ tests/
├─ artifacts/
├─ reports/
├─ pyproject.toml
└─ README.md
>>>>>>> 20a6f7f (Update README)

````

---

## How to Run

<<<<<<< HEAD
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
=======
### Setup
```bash
pip install -r requirements.txt
````

---

### Milestone 1 Pipeline

```bash
python -m scripts.run_data_ingest
python -m scripts.run_pair_gen
python -m scripts.run_benchmark
```

Or full pipeline:

```bash
python -m scripts.run_pipeline
```

---

### Milestone 2 Pipeline

#### Baseline

```bash
python -m scripts.run_val_sweep
python -m scripts.run_val_eval
python -m scripts.run_baseline_test_eval
```

#### Data-Centric Improvement

```bash
python -m scripts.sample_validation_pairs
python -m scripts.run_sampled_val_sweep
python -m scripts.run_sampled_val_eval
python -m scripts.run_sampled_test_eval
```

---

### Run Tests

```bash
pytest
```

---

## Outputs

* `artifacts/train_pairs.npy`, `train_labels.npy`
* `artifacts/val_pairs.npy`, `val_labels.npy`
* `artifacts/test_pairs.npy`, `test_labels.npy`
* `artifacts/runs/` → tracked experiment runs
* threshold sweep and best-threshold JSON files
* benchmarking outputs

---

## Experiment Tracking

Each run records:

* run_id
* timestamp
* code version
* split (val/test)
* data version
* threshold rule
* selected threshold
* metrics
* notes

### Key Runs

* `baseline_val_sweep`
* `baseline_val_selected`
* `baseline_test_eval`
* `sampled_val_sweep`
* `sampled_val_eval_best`
* `sampled_test_eval`

---

## Threshold Selection
>>>>>>> 20a6f7f (Update README)

* Rule: **maximize balanced accuracy on validation**
* Selected on validation only (no test leakage)
* Fixed before test evaluation

<<<<<<< HEAD
Saved automatically in `artifacts/`:

* `dataset_manifest.json` – Dataset info and split sizes.
* `train_pairs.npy`, `train_labels.npy` – Training pairs and labels.
* `val_pairs.npy`, `val_labels.npy` – Validation pairs and labels.
* `test_pairs.npy`, `test_labels.npy` – Test pairs and labels.

Benchmark prints:
=======
---

## Data-Centric Change
>>>>>>> 20a6f7f (Update README)

Problem:

<<<<<<< HEAD
```
=======
* Validation set heavily dominated by negative pairs

Solution:

* Deterministic sampling:

  * keep all positives
  * sample negatives to 3:1 ratio

Effect:

* Preserves ROC behavior
* Reduces computation
* Produces same optimal threshold

---

## Results (Test Set)

* Precision ≈ 0.003
* Recall ≈ 0.36
* Balanced Accuracy ≈ 0.53

Observation:

* High false positives
* Model struggles with visually similar faces

---

## Error Analysis

**False Negatives**

* Same person under different conditions (pose, lighting, expression)
* Cause: sensitivity to intra-class variation

**False Positives**

* Different individuals with similar appearance
* Cause: weak feature discrimination

---

## Validation & Reliability

The pipeline includes:

* input validation (pairs, labels, splits)
* threshold validation
* score consistency checks
* metric validation
* fail-fast error handling

---

## Tests

* Unit tests (metrics, threshold logic, validation)
* Integration test (end-to-end pipeline check)

Run:

```bash
pytest
```

---

## Milestone 2 Report

```
reports/milestone2_report.pdf
```

Includes:

* ROC curve
* confusion matrix
* error slices
* baseline vs improved comparison

---

## Reproducibility

From a clean clone:

1. install requirements
2. run commands above
3. results and runs will reproduce

Final reported result:

* run: `sampled_test_eval`
* threshold: 0.76

---
>>>>>>> 20a6f7f (Update README)


<<<<<<< HEAD
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
=======
All data processing steps are deterministic:

* fixed random seed
* reproducible pair generation
* identical outputs across runs

---

## Submission

* Git tag: `v0.2`
* Includes reproducible pipeline, report, tests, and tracked runs

```
```
>>>>>>> 20a6f7f (Update README)
