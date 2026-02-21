# Face Verification Project

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

1. **Install required packages**  

pip install tensorflow-datasets numpy


2. **Run data ingestion**
   Generates LFW splits and saves dataset manifest.

python src/data_ingest.py


3. **Generate image pairs**
   Creates positive and negative pairs and saves them in `artifacts/`.

python src/pair_gen.py


4. **Run similarity benchmark**
   Compare Python loop vs NumPy implementations for cosine similarity and Euclidean distance.

python src/benchmark.py


Full Pipeline

Run everything in one go using:

python -m scripts.run_pipeline

This will ingest the dataset, generate pairs and run the benchmark automatically. You should see runtime comparisons and correctness verification in the output.

## Output

* `artifacts/dataset_manifest.json` – Dataset info and split sizes
* `artifacts/train_pairs.npy`, `train_labels.npy` – Training pairs and labels
* `artifacts/val_pairs.npy`, `val_labels.npy` – Validation pairs and labels
* `artifacts/test_pairs.npy`, `test_labels.npy` – Test pairs and labels

The benchmark prints:

* Time taken for Python loops and NumPy vectorized operations.
* Speedup factor of NumPy vs loops.
* Correctness check confirmation for both similarity measures.


