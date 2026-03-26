# Loads the best threshold found from the full validation sweep and uses it to evaluate the test set
import os
import numpy as np
import json

from src.config import OUTPUT_DIR, SCORE_FUNCTION
from src.evaluation import load_lfw_images, evaluate_pairs
from src.run_tracker import save_run
from src.validation import (
    validate_split_name,
    validate_pairs,
    validate_labels,
    validate_pairs_and_labels_match,
    validate_threshold,
    validate_metrics,
)


def main():
    # File containing the threshold selected from the baseline validation sweep
    threshold_source = "val_best_threshold.json"
    best_path = os.path.join(OUTPUT_DIR, threshold_source)

    # Raise error if the threshold artifact is missing
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing file: {best_path}")

    # Prevent accidental data leakage
    if "test" in threshold_source.lower():
        raise ValueError("Threshold source must not come from test artifacts")

    # Load the selected threshold
    with open(best_path, "r") as f:
        best = json.load(f)

    selected_threshold = best["threshold"]
    validate_threshold(selected_threshold)

    # Evaluate the held-out test split
    split_name = "test"
    validate_split_name(split_name)

    # Paths to deterministic test pair files
    pairs_path = os.path.join(OUTPUT_DIR, f"{split_name}_pairs.npy")
    labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.npy")

    # Fail early if required data artifacts are missing
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing file: {pairs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    # Load deterministic test pairs and labels
    pairs = np.load(pairs_path)
    labels = np.load(labels_path).astype(int)

    # Validate inputs
    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)

    # Load all LFW images
    print("Loading images...")
    images = load_lfw_images()

    # Run evaluation using the selected baseline validation threshold
    results = evaluate_pairs(
        images=images,
        pairs=pairs,
        labels=labels,
        threshold=selected_threshold,
        score_type=SCORE_FUNCTION,
    )

    # Validate metrics output structure
    validate_metrics(results["metrics"])

    # Save the tracked baseline final test run
    save_run(
        run_id="baseline_test_eval",
        split="test",
        data_version="test_pairs.npy",
        score_function=SCORE_FUNCTION,
        threshold_rule="maximize balanced_accuracy on validation",
        selected_threshold=selected_threshold,
        metrics=results["metrics"],
        note="Final test evaluation using threshold selected from full validation sweep",
        extra={
            "threshold_source": threshold_source,
            "num_pairs": int(len(pairs)),
        },
    )

    # Print results
    print(f"Split: {split_name}")
    print(f"Score type: {results['score_type']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Number of pairs: {results['num_pairs']}")
    print("Metrics:")

    # Print results
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()