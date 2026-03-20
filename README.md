# Face Verification Project

**Group:** ML Verifier  
**Group Members:** Moataz Osama Saadeldin Abdelaziz, Ankan Roy

---

## Overview

This project implements a face verification system using the **Labeled Faces in the Wild (LFW)** dataset.

The system takes two face images and outputs:
- a similarity score  
- a same-person vs different-person decision based on a threshold  

The project is structured in two milestones:

- **Milestone 1:** deterministic pipeline (data ingestion, pair generation, similarity scoring, benchmarking)  
- **Milestone 2:** reproducible evaluation system with threshold calibration, experiment tracking, data-centric iteration, and error analysis  

---

## Milestone 1 (Baseline)

- Deterministic LFW ingestion and dataset split  
- Deterministic pair generation (train / validation / test)  
- Similarity computation:
  - Cosine similarity  
  - Euclidean distance  
- Benchmarking Python loops vs NumPy vectorization  

---

## Milestone 2 (Evaluation & Iteration)

### Baseline
- Threshold selected from full validation set  
- Rule: **maximize balanced accuracy**  
- Final evaluation on held-out test set  

### Data-Centric Improvement
- Keep all positive pairs  
- Sample negatives to **3:1 ratio (negative:positive)**  
- Threshold re-selected on sampled validation  
- Test set unchanged  

---

## Key Result

Both baseline and improved systems produced the same threshold:

> **Threshold = 0.76**

This indicates:
- stable similarity ranking  
- robust threshold selection  

---

## Project Structure

```text
face-verification/
│
├─ src/
│  ├─ data_ingest.py
│  ├─ pair_gen.py
│  ├─ similarity.py
│  ├─ benchmark.py
│  ├─ evaluation.py
│  ├─ metrics.py
│  ├─ validation.py
│  ├─ run_tracker.py
│  └─ config.py
│
├─ scripts/
│  ├─ run_pipeline.py
│  ├─ run_data_ingest.py
│  ├─ run_pair_gen.py
│  ├─ run_benchmark.py
│  ├─ run_val_sweep.py
│  ├─ run_val_eval.py
│  ├─ run_baseline_test_eval.py
│  ├─ sample_validation_pairs.py
│  ├─ run_sampled_val_sweep.py
│  ├─ run_sampled_val_eval.py
│  └─ run_sampled_test_eval.py
│
├─ tests/
├─ artifacts/
├─ reports/
├─ pyproject.toml
├─ requirements.txt
└─ README.md
````

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run full pipeline

```bash
python -m scripts.run_pipeline
```

---

## Milestone 2 Pipeline

### Baseline

```bash
python -m scripts.run_val_sweep
python -m scripts.run_val_eval
python -m scripts.run_baseline_test_eval
```

### Data-Centric Improvement

```bash
python -m scripts.sample_validation_pairs
python -m scripts.run_sampled_val_sweep
python -m scripts.run_sampled_val_eval
python -m scripts.run_sampled_test_eval
```

---

## Run Tests

```bash
pytest
```

---

## Outputs

* `artifacts/train_pairs.npy`, `train_labels.npy`
* `artifacts/val_pairs.npy`, `val_labels.npy`
* `artifacts/test_pairs.npy`, `test_labels.npy`
* experiment tracking logs in `artifacts/runs/`
* threshold sweep outputs

---

## Experiment Tracking

Each run records:

* run_id
* timestamp
* code version
* dataset split
* selected threshold
* evaluation metrics

---

## Threshold Selection

* Rule: **maximize balanced accuracy on validation set**
* No test leakage
* Fixed before test evaluation

---

## Data-Centric Change

**Problem:**

* Validation set dominated by negative pairs

**Solution:**

* Deterministic sampling
* Keep all positives
* Sample negatives to **3:1 ratio**

**Effect:**

* Preserves ROC behavior
* Reduces computation
* Produces same optimal threshold

---

## Evaluation Pipeline

```text
baseline_val_sweep
        ↓
sample_validation_pairs
        ↓
sampled_val_sweep
        ↓
best_threshold_selection
        ↓
sampled_test_eval
```

---

## Results (Test Set)

* Precision ≈ 0.003
* Recall ≈ 0.36
* Balanced Accuracy ≈ 0.53

**Observation:**

* High false positives
* Difficulty distinguishing visually similar faces

---

## Error Analysis

### False Negatives

* Same person under different conditions
* Cause: intra-class variation

### False Positives

* Different people with similar appearance
* Cause: weak feature discrimination

---

## Validation & Reliability

* Input validation (pairs, labels, splits)
* Threshold validation
* Metric consistency checks
* Fail-fast error handling

---

## Milestone 2 Report

Located at:

```
reports/milestone2_report.pdf
```

Includes:

* ROC curve
* confusion matrix
* error analysis
* baseline vs improved comparison

---

## Reproducibility

All steps are deterministic:

* fixed random seed
* reproducible pair generation
* identical outputs across runs

From a clean clone:

1. install dependencies
2. run pipeline commands
3. results will reproduce

