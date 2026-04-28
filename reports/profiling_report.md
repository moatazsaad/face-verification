# Profiling Report

## Measurement Environment

This report provides the required CPU baseline profiling for the final face verification system.

- Hardware: CPU baseline
- CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- Model: InsightFace `buffalo_s`
- Score type: `cosine_with_embeddings`
- Operating threshold: `0.29`
- Profiling script: `scripts/run_profiling.py`
- Profiling output file: `reports/profiling_cpu_summary.csv`

## Methodology

The profiling script reused the final inference path from `src/run_inference_cli.py`. This keeps the profiling results aligned with the same system used by the CLI.

The profiling command was:

```bash
python -m scripts.run_profiling
````

The script measured:

* preprocessing latency
* embedding-generation latency
* scoring latency
* total inference latency
* p95 total latency
* throughput in pairs per second

Batch size means the number of image-pair inference requests processed in one profiling group. Each request compares two images. Each batch size was repeated 3 times.

## CPU Profiling Results

| Batch Size | Repeats | Total Requests | Preprocessing Mean (ms) | Embedding Mean (ms) | Scoring Mean (ms) | Total Mean (ms) | Total P95 (ms) | Throughput (pairs/sec) |
| ---------: | ------: | -------------: | ----------------------: | ------------------: | ----------------: | --------------: | -------------: | ---------------------: |
|          1 |       3 |              3 |                   0.012 |             293.585 |             0.021 |         293.618 |        135.736 |                  0.738 |
|          2 |       3 |              6 |                   0.009 |             112.017 |             0.013 |         112.039 |        130.111 |                  8.777 |
|          4 |       3 |             12 |                   0.009 |              89.653 |             0.013 |          89.675 |        120.193 |                 10.931 |
|          8 |       3 |             24 |                   0.008 |              72.065 |             0.013 |          72.086 |        116.722 |                 13.473 |

## Interpretation

The embedding stage dominates runtime. This is expected because embedding generation includes face detection and feature extraction using the InsightFace `buffalo_s` model.

Preprocessing time is very small because it mainly prepares the image format before embedding extraction. Scoring time is also very small because cosine similarity only compares two embedding vectors.

The batch-size results show that throughput improves as the batch size increases. Throughput increases from 0.738 pairs per second at batch size 1 to 13.473 pairs per second at batch size 8. This suggests that repeated inference runs reduce the impact of overhead and make better use of the already loaded model.

The total mean latency also decreases across larger batch sizes, from 293.618 ms at batch size 1 to 72.086 ms at batch size 8. This supports the conclusion that the main runtime cost is embedding generation, while preprocessing and scoring are minimal.

The p95 latency values provide a simple estimate of slower inference runs. Because each batch size was repeated only 3 times, the p95 values should be treated as a small local profiling estimate rather than a large-scale benchmark.

## Limitations

The profiling results are based on one CPU environment and may not generalize to other hardware configurations.

No GPU profiling was included, so this report does not measure possible GPU speedups.

The profiling uses a small set of repeated pair-level inference runs. It is sufficient for a local CPU baseline and batch-size sensitivity analysis, but it should not be interpreted as a large-scale production benchmark.

## Summary

The final system was profiled on CPU and produced a clear latency breakdown for preprocessing, embedding generation, scoring, and total inference time. The results show that embedding generation is the main runtime cost, while preprocessing and scoring are very small. The batch-size comparison provides the required batch-size sensitivity analysis for the final release.

```

