Here is your **complete final README (clean, concise, and rubric-perfect)** with everything integrated properly:

---

````markdown
# Face Verification Project

**Group:** ML Verifier  
**Group Members:** Moataz Osama Saadeldin Abdelaziz, Ankan Roy  

---

## Overview

This project implements a face verification system using the **Labeled Faces in the Wild (LFW)** dataset.

The system takes two face images and outputs:
- a similarity score  
- a same-person vs different-person decision using a threshold  

The project evolves from a deterministic pipeline (Milestone 1) into a **reproducible ML evaluation system** (Milestone 2) with:
- experiment tracking  
- threshold calibration  
- data-centric iteration  
- error analysis  

---

## Milestone 1 (Baseline System)

- Deterministic dataset ingestion (LFW via TFDS)  
- Deterministic train / validation / test split  
- Deterministic pair generation  
- Similarity computation:
  - cosine similarity  
  - euclidean distance  
- Benchmarking: Python loops vs NumPy  

---

## Milestone 2 (Evaluation System)

### Baseline
- Threshold selected on full validation set  
- Rule: **maximize balanced accuracy**  
- Final evaluation on held-out test set  

### Data-Centric Improvement
- Keep all positive pairs  
- Sample negative pairs to **3:1 ratio (negative:positive)**  
- Implemented in: `src/validation_sampling.py`  
- Re-run threshold selection on sampled validation  
- Test set remains unchanged  

---

## Key Result

Both baseline and improved systems produced:

> **Selected threshold = 0.76**

This indicates:
- stable similarity score ranking  
- threshold selection is robust to validation sampling  

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
│  ├─ validation_sampling.py   # negative sampling (3:1)
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
│   └─ runs/
├─ reports/
│   └─ milestone2_report.pdf
│
├─ requirements.txt
├─ pyproject.toml
└─ README.md
````

---

## How to Run (Clean Reproduction)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Generate dataset and pairs (Milestone 1)

```bash
python -m scripts.run_data_ingest
python -m scripts.run_pair_gen
```

---

### 3. Baseline evaluation (Milestone 2)

```bash
python -m scripts.run_val_sweep
python -m scripts.run_val_eval
python -m scripts.run_baseline_test_eval
```

---

### 4. Data-centric improvement

```bash
python -m scripts.sample_validation_pairs
python -m scripts.run_sampled_val_sweep
python -m scripts.run_sampled_val_eval
python -m scripts.run_sampled_test_eval
```

---

### 5. Run tests

```bash
pytest
```

---

## Outputs

Generated in `artifacts/`:

* `train_pairs.npy`, `train_labels.npy`
* `val_pairs.npy`, `val_labels.npy`
* `val_pairs_sampled.npy`, `val_labels_sampled.npy`
* `test_pairs.npy`, `test_labels.npy`

Experiment tracking:

* `artifacts/runs/*.json`

Plots:

* ROC curves
* confusion matrix

---

## Experiment Tracking

Each run logs:

* run_id
* timestamp
* code version
* dataset split
* selected threshold
* evaluation metrics

Minimum required tracked runs included:

* baseline validation sweep
* baseline validation evaluation
* baseline test evaluation
* sampled validation sweep
* sampled test evaluation

---

## Threshold Selection

* Rule: **maximize balanced accuracy on validation**
* Selected on validation only
* Fixed before test evaluation
* No data leakage

---

## Data-Centric Change

**Problem:**
Validation set heavily imbalanced (too many negative pairs)

**Solution:**

* Keep all positives
* Sample negatives to 3:1 ratio
* Deterministic sampling using fixed seed

**Effect:**

* Preserves ROC behavior
* Reduces computation
* Produces same optimal threshold

---

## Evaluation Pipeline

```text
baseline_val_sweep
        ↓
baseline_test_eval
        ↓
validation_sampling (3:1)
        ↓
sampled_val_sweep
        ↓
sampled_test_eval
```

---

## Final Test Results

* Precision ≈ 0.003
* Recall ≈ 0.36
* Balanced Accuracy ≈ 0.53

**Observation:**

* high false positives
* difficulty distinguishing similar faces

---

## Error Analysis

### False Negatives (Same Person)

* variation in pose, lighting, expression
* model sensitive to intra-class variation

### False Positives (Different People)

* visually similar individuals
* model relies on coarse features

---

## Validation & Reliability

* input validation (pairs, labels, splits)
* threshold validation
* metric consistency checks
* fail-fast errors

---

## Tests

Includes:

* unit tests:

  * metrics
  * threshold logic
  * validation checks

* integration test:

  * end-to-end pipeline on small synthetic data

---

## Report

Located at:

```
reports/milestone2_report.pdf
```

Includes:

* ROC curve
* confusion matrix
* baseline vs improved comparison
* error slices

---

## Reproducibility

The pipeline is fully deterministic:

* fixed random seed
* deterministic pair generation
* reproducible threshold selection

From a clean clone:

1. install dependencies
2. run pipeline commands
3. results reproduce exactly




