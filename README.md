# Face Verification Project

<<<<<<< HEAD
**Group:** ML Verifier  
**Group Members:** Moataz Osama Saadeldin Abdelaziz, Ankan Roy  
=======
Group: ML Verifier

Group Members: Moataz Osama Saadeldin Abdelaziz, Ankan Roy

## Overview
This project implements a face verification system using the Labeled Faces in the Wild (LFW) dataset. It includes:
>>>>>>> origin/main

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

**Committed**
```
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
│ ├─ run_pipeline.py       # Runs the full pipeline: ingestion, pair generation, benchmarking
| ├─ run_data_ingest.py    # Runs data ingestion, creates manifest
| ├─ run_pair_gen.py       # Runs image pair generation
| └─ run_benchmark.py      # Runs similarity module
│
├─ tests/
├─ artifacts/
│   └─ runs/
├─ reports/
│   └─ milestone2_report.pdf
│
├─ artifacts/              # Output directory for manifest, pairs and labels
├─ pyproject.toml          # Project metadata and dependencies
└─ README.md               # This file
```

**Ignored**
```
face-verification/
│
├─ data/                   # Stores the LFW build
├─ artifacts/              # Stores outputs from pair generation and benchmarking

````

---

This project uses the LFW dataset via TensorFlow Datasets (TFDS).

The code loads:
    lfw:0.1.1

Make sure the dataset is already downloaded in your local TFDS cache before running the pipeline.

**Running Full Pipeline**

Install Packages:

`pip install -r requirements.txt`

Running Individual Scripts:

`python -m scripts.run_data_ingest`

`python -m scripts.run_pair_gen`

`python -m scripts.run_benchmark`

Run everything in one go using:

`python -m scripts.run_pipeline`

This will ingest the dataset, generate pairs and run the benchmark automatically. You should see runtime comparisons and correctness verification in the output.

NOTE: The program takes 3-4 minutes to run, while unique deterministic pairs are being generated.

**Using Linux/MacOS**

Create Virtual Environment:

`python3 -m venv tf_env`

`source tf_env/bin/activate`

`pip install -r requirements.txt`

Running Individual Scripts:

`python3 -m scripts.run_data_ingest`

`python3 -m scripts.run_pair_gen`

`python3 -m scripts.run_benchmark`

Run Full Script:

`python3 -m scripts.run_pipeline`


## Output

* `artifacts/dataset_manifest.json` – Dataset info and split sizes
* `artifacts/train_pairs.npy`, `train_labels.npy` – Training pairs and labels
* `artifacts/val_pairs.npy`, `val_labels.npy` – Validation pairs and labels
* `artifacts/test_pairs.npy`, `test_labels.npy` – Test pairs and labels
* `artifacts/cosine_loop_times.npy`, `cosine_numpy_times.npy` - Cosine similarity benchmarking
* `artifacts/euclidean_loop_times.npy`, `euclidean_numpy_times.npy` - Euclidean distance benchmarking

The benchmark prints:

* Time taken for Python loops and NumPy vectorized operations.
* Speedup factor of NumPy vs loops.
* Correctness check confirmation for both similarity measures.


## Determinism Notes

The data ingestion and pair generation steps are deterministic, and the seed can be found in the data manifest. They set a fixed random seed at the start of execution. By initializing the random number generator with the same constant seed each time, any operations that rely on randomness, such as shuffling or pairing, will produce the same results across runs. So long as the input data remains unchanged and the seed value is fixed, the output will always be identical.

A few simple tests can be run to check determinism:

`python3 -m tests.test_data_ingest`

`python3 -m tests.test_pair_generation`
