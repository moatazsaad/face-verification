# Reproducibility Checklist

### Final Git Tag

Final tag: `v1.0-final`

This tag points to the final reproducible release for Milestone 4.

Follow the steps below to reproduce the core results of the face verification system.

### Clone Repository and Set Up Environment

```bash
git clone <repository-url>
cd face-verification
python3 -m venv tf_env
source tf_env/bin/activate
pip install -r requirements.txt
```

### Run CLI Inference (Single Pair)

```bash
python -m src.run_inference_cli --image1 examples/sample1.jpg --image2 examples/sample2.jpg
```

### Run Batch Inference

Build the Docker image:

```bash
python -m src.run_inference_cli --folder examples/
```

### Run Milestone 4 CPU Profiling

```bash
python -m scripts.run_profiling
```

Expected output file:

```text
reports/profiling_cpu_summary.csv
```

### Run Docker

Build the Docker image:

```bash
docker build -t face-verification .
```

Run CLI inference inside Docker:

```bash
docker run --rm face-verification --image1 "/app/examples/sample1.jpg" --image2 "/app/examples/sample2.jpg"
```

Expected output includes:

* pair ID
* similarity score
* score type
* threshold
* match/non-match decision
* confidence
* preprocessing time
* embedding time
* scoring time
* total inference time

### Run Tests

```bash
python -m pytest
```

### Locate Final Artifacts

* System Card: `reports/`
* Profiling Report: `reports/`
* Example inputs: `examples/`

### Final Release Version

```bash
git checkout v1.0-final
```

## Final System Settings

* Embedding model: InsightFace `buffalo_s`
* Score type: `cosine_with_embeddings`
* Similarity metric: cosine similarity
* Operating threshold: `0.29`
* Interface: CLI
* CPU profiling artifact: `reports/profiling_cpu_summary.csv`

## Final Release Artifacts

* README: `README.md`
* System Card: `reports/system_card.md`
* Profiling Report: `reports/profiling_report.md`
* Profiling CSV: `reports/profiling_cpu_summary.csv`
* Reproducibility Checklist: `reports/reproducibility_checklist.md`
* Final configuration: `src/config.py`
