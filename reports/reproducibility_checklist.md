# Milestone 4 Reproducibility Checklist

## Final Git Tag

Final tag: `v1.0-final`

This tag points to the final reproducible release for Milestone 4.

## Environment Setup

From the project root:

```bash
python -m venv face-verification
face-verification\Scripts\activate
pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest
```

## Run CLI Inference

```bash
python -m src.run_inference_cli --image1 examples/sample1.jpg --image2 examples/sample2.jpg
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

## Run Milestone 4 CPU Profiling

```bash
python -m scripts.run_profiling
```

Expected output file:

```text
reports/profiling_cpu_summary.csv
```

This file includes:

* CPU information
* batch size
* total requests
* preprocessing mean latency
* embedding mean latency
* scoring mean latency
* total mean latency
* p95 latency
* throughput

## Run Docker

Build the Docker image:

```bash
docker build -t face-verification .
```

Run CLI inference inside Docker:

```bash
docker run --rm face-verification --image1 "/app/examples/sample1.jpg" --image2 "/app/examples/sample2.jpg"
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


