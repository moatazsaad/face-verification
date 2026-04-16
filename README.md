# Face Verification Project

**Group:** ML Verifier  
**Group Members:** Moataz Osama Saadeldin Abdelaziz, Ankan Roy  

---

## Overview

This project implements a face verification system using the Labeled Faces in the Wild (LFW) dataset.

- **Milestone 1:** deterministic pipeline (data ingestion, pair generation, similarity scoring, benchmarking)  
- **Milestone 2:** reproducible evaluation system with threshold calibration, experiment tracking, data-centric iteration, and error analysis 
- **Milestone 3:** uses embeddings to compare two images, and extends it with a CLI interface, Docker packaging, confidence calibration, and concurrent load testing

The system takes two face images and outputs:
- a similarity score  
- a same-person vs different-person decision based on a threshold  

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
│ ├─ generate_image_embeddings.py
│ ├─ run_inference_cli.py
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
│ ├─ run_load_test.py
│ └─ run_sampled_test_eval.py
│
├─ tests/
├─ artifacts/
├─ reports/
├─ pyproject.toml
└─ README.md

````

---

### Setup
```bash
pip install -r requirements.txt
````

### Milestone 1 Pipeline

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

### Milestone 2 Pipeline

#### Baseline

```bash
python -m scripts.run_baseline_val_sweep
python -m scripts.run_baseline_val_eval
python -m scripts.run_baseline_test_eval
```

#### Data-Centric

```bash
python -m scripts.run_validation_sampling
python -m scripts.run_sampled_val_sweep
python -m scripts.run_sampled_val_eval
python -m scripts.run_sampled_test_eval
```

---

## Milestone 3

This project implements a face verification system using deep embeddings. Given two input images, the system determines whether they belong to the same person by computing a similarity score between their embeddings.

Milestone 3 extends the system by adding a clean CLI inference interface, Docker packaging, confidence calibration, and a concurrent load testing setup.

---

## Pipeline Summary

The inference pipeline consists of:

1. **Preprocessing**
   Load image and convert to RGB format.

2. **Embedding Generation**
   Extract face embeddings using InsightFace (`buffalo_s` model).

3. **Similarity Scoring**
   Compute cosine similarity between embeddings.

4. **Threshold Decision**
   Compare score to a fixed threshold (`0.29`) to determine match.

5. **Confidence Computation**
   Compute a margin-based confidence score based on distance from threshold.

6. **Latency Measurement**
   Measure total inference time per request.

---

### How to Run

**Run CLI Inference:**

```bash
python -m src.run_inference_cli --image1 examples/sample1.jpg --image2 examples/sample2.jpg
```

**Example CLI Output:**

```text
Inference Result
------------------
pair_id=(examples/sample1.jpg, examples/sample2.jpg)
score=-0.10245499759912491
score_type=cosine_with_embeddings
threshold=0.29
decision=non-match
confidence=1.0
preprocessing_time_ms=0.02
embedding_time_ms=268.44
score_time_ms=0.02
total_time_ms=268.48
```

**Run Load Test (Concurrency):**

```bash
python -m scripts.run_load_test --pairs_file examples/load_test_pairs.json --requests 10 --workers 3
```

**Run Tests:**

```bash
python -m pytest
```

**Docker:**

Build:

```bash
docker build -t face-verification .
```

Run:

```bash
docker run --rm face-verification --image1 "/app/examples/sample1.jpg" --image2 "/app/examples/sample2.jpg"
```

**Artifact Locations:**

Example images:

```bash
examples/
```

Load test input pairs:

```bash
examples/load_test_pairs.json
```

Load test output:

printed in terminal summary

**Embeddings:**

The embedding module, generate_image_embeddings.py, first precomputes the embeddings of each image in our chosen split, in this case, the validation split. Afterwards, we have a function to compute the cosine similarity or Euclidean distance of the image pairs based on these embeddings. After running our threshold sweeps on these embeddings, we have the following results:

Baseline Validation Sweep’s Selected Threshold: 0.35

Sampled Validation Sweep’s Selected Threshold: 0.29

**Confidence:**

The CLI reports a confidence value between 0 and 1. This confidence is not a probability, but rather a simple margin-based score that shows how far the similarity score is from the operating threshold.

For the embedding-based cosine system:

•	Higher confidence means the score is farther from the threshold, so the decision is more clear.

•	Lower confidence means the score is closer to the threshold, so the decision is less certain.

Confidence is computed as:

```bash
confidence = clip(abs(score - threshold) / margin_scale, 0, 1)
```

Interpretation:

•	Confidence near 0: score is very close to the decision boundary

•	Confidence near 1: score is far from the decision boundary

Example:

•	If the score is 0.30, confidence is low because it is close to the threshold 0.29

•	If the score is 0.95 or -0.10, confidence is high because it is far from the threshold

**Design Notes:**

•	Embedding model: InsightFace buffalo_s (fast and lightweight)

•	Similarity metric: cosine similarity on normalized embeddings

•	Threshold: fixed at 0.29 based on sampled validation sweep

•	Confidence: margin-based, derived from distance to threshold

•	Inference interface: CLI using argparse

•	Load testing: concurrent execution using ThreadPoolExecutor



