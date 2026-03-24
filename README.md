```markdown
# Face Verification Project

**Group:** ML Verifier  
**Group Members:** Moataz Osama Saadeldin Abdelaziz, Ankan Roy  

---

## Overview

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

## Project Structure

**Committed**
```

face-verification/
│
├─ src/
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
│ ├─ run_baseline_val_sweep.py
│ ├─ run_baseline_val_eval.py
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

````

---

## How to Run

### Setup
```bash
pip install -r requirements.txt
````

---

### Milestone 1 Pipeline

Create virtual environment:
```bash
python3 -m venv tf_env
source tf_env/bin/activate
pip install -r requirements.txt
```

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
python -m scripts.run_baseline_val_sweep
python -m scripts.run_baseline_val_eval
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

* Rule: **maximize balanced accuracy on validation**
* Selected on validation only (no test leakage)
* Fixed before test evaluation

---

## Data-Centric Change

Problem:

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

## Determinism Notes

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
