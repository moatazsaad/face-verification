import argparse
import csv
import platform
import time
from pathlib import Path

from src.run_inference_cli import run_single_inference


def average(values):
    # Avoid division by zero if the list is empty
    return sum(values) / len(values) if values else 0


def p95(values):
    # p95 shows tail latency: most requests are faster than this value
    if not values:
        return 0

    values = sorted(values)
    return values[int(0.95 * (len(values) - 1))]


def main():
    parser = argparse.ArgumentParser(description="Milestone 4 CPU profiling script")

    # Default sample images used for profiling
    parser.add_argument("--image1", default="examples/sample1.jpg")
    parser.add_argument("--image2", default="examples/sample2.jpg")

    # Batch sizes are used to show batch-size sensitivity
    parser.add_argument("--batch_sizes", default="1,2,4,8")

    # Repeats make the average more stable
    parser.add_argument("--repeats", type=int, default=3)

    # Output CSV file for the profiling artifact
    parser.add_argument("--output", default="reports/profiling_cpu_summary.csv")

    args = parser.parse_args()

    image1_path = Path(args.image1)
    image2_path = Path(args.image2)
    output_path = Path(args.output)

    # Fail early if the sample images are missing
    if not image1_path.exists():
        raise FileNotFoundError(f"Image 1 not found: {image1_path}")

    if not image2_path.exists():
        raise FileNotFoundError(f"Image 2 not found: {image2_path}")

    # Create reports/ folder if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    rows = []

    for batch_size in batch_sizes:
        results = []

        # Wall time is used for throughput calculation
        wall_start = time.perf_counter()

        for repeat in range(args.repeats):
            for i in range(batch_size):
                # Reuse the existing Milestone 3 inference function.
                # This keeps profiling aligned with the real CLI inference logic.
                result = run_single_inference(
                    args.image1,
                    args.image2,
                    pair_id=f"batch_{batch_size}_repeat_{repeat}_pair_{i}",
                )
                results.append(result)

        wall_time = time.perf_counter() - wall_start

        # These stage times come directly from run_single_inference()
        preprocessing_times = [r["preprocessing_time_ms"] for r in results]
        embedding_times = [r["embedding_time_ms"] for r in results]
        scoring_times = [r["score_time_ms"] for r in results]
        total_times = [r["total_time_ms"] for r in results]

        # One row per batch size for easy comparison
        row = {
            "batch_size": batch_size,
            "repeats": args.repeats,
            "total_requests": len(results),
            "cpu": platform.processor(),
            "score_type": results[0]["score_type"],
            "threshold": round(results[0]["threshold"], 3),
            "preprocessing_mean_ms": round(average(preprocessing_times), 3),
            "embedding_mean_ms": round(average(embedding_times), 3),
            "scoring_mean_ms": round(average(scoring_times), 3),
            "total_mean_ms": round(average(total_times), 3),
            "total_p95_ms": round(p95(total_times), 3),
            "throughput_pairs_per_sec": round(
                len(results) / wall_time if wall_time > 0 else 0,
                3,
            ),
        }

        rows.append(row)
        print(row)

    # Save profiling results as a CSV artifact for Milestone 4
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved profiling results to: {output_path}")


if __name__ == "__main__":
    main()