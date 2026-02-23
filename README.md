# Face Verification Project

Group: ML Verifier
Group Members: Moataz Osama Saadeldin Abdelaziz, Ankan Roy

## Overview
This project implements a face verification system using the Labeled Faces in the Wild (LFW) dataset. It includes:

- Image ingestion and train/validation/test split.
- Generation of positive and negative image pairs for verification.
- Similarity calculations using Cosine Similarity and Euclidean Distance.
- Comparison of Python loop vs vectorized NumPy implementations for performance benchmarking and correctness checks.

## Project Structure

```

face-verification/
│
├─ src/
│ ├─ data_ingest.py        # Load and split LFW dataset
│ ├─ pair_gen.py           # Generate positive and negative image pairs
│ ├─ similarity.py         # Similarity functions (Python loop & NumPy)
│ ├─ benchmark.py          # Benchmark script comparing Python loops vs NumPy
│ └─ config.py             # Configuration file for seeds, ratios and output directory
│
├─ scripts/
│ └─ run_pipeline.py       # Runs the full pipeline: ingestion, pair generation, benchmarking
│
├─ notebooks/
│ └─ notebook.ipynb          
│
├─ artifacts/              # Output directory for manifest, pairs and labels
├─ pyproject.toml          # Project metadata and dependencies
└─ README.md               # This file

````

## How to Run


**Running Full Pipeline**

Install Packages:

`pip install tensorflow-datasets numpy`

Run everything in one go using:

`python -m scripts.run_pipeline`

This will ingest the dataset, generate pairs and run the benchmark automatically. You should see runtime comparisons and correctness verification in the output.

**Using Mac**

Create Virtual Environment:

`python3 -m venv tf_env`

`source tf_env/bin/activate`

`python -m pip install tensorflow tensorflow-datasets numpy`

Run Full Script:

`python3 -m scripts.run_pipeline`


## Output

* `artifacts/dataset_manifest.json` – Dataset info and split sizes
* `artifacts/train_pairs.npy`, `train_labels.npy` – Training pairs and labels
* `artifacts/val_pairs.npy`, `val_labels.npy` – Validation pairs and labels
* `artifacts/test_pairs.npy`, `test_labels.npy` – Test pairs and labels

The benchmark prints:

* Time taken for Python loops and NumPy vectorized operations.
* Speedup factor of NumPy vs loops.
* Correctness check confirmation for both similarity measures.


