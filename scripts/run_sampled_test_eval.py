import os
import numpy as np
import json
from src.config import (
    OUTPUT_DIR, SCORE_FUNCTION, VAL_NEGATIVE_RATIO,
    sampled_best_threshold_filename, sampled_pairs_filename
)
from src.evaluation import load_lfw_images, evaluate_pairs
from src.run_tracker import save_run
from src.validation import (
    validate_split_name, validate_pairs, validate_labels,
    validate_pairs_and_labels_match, validate_threshold, validate_metrics
)

def main():
    # Ratio tag used for saving filename
    ratio_tag = f"neg{VAL_NEGATIVE_RATIO}x"

    # File containing the threshold selected from the baseline validation sweep
    threshold_source = sampled_best_threshold_filename()
    best_path = os.path.join(OUTPUT_DIR, threshold_source)

    # Raise error if the threshold artifact is missing
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing file: {best_path}")

    # Prevent accidental data leakage
    if "test" in threshold_source.lower():
        raise ValueError("Threshold source must not come from test artifacts")

    with open(best_path, "r") as f:
        best = json.load(f)

    # Get the selected threshold
    selected_threshold = best["threshold"]

    # Validate that the threshold is within an acceptable numeric range
    validate_threshold(selected_threshold)

    # This script evaluates the held-out test split
    split_name = "test"
    validate_split_name(split_name)

    # Load pairs and labels
    pairs_path = os.path.join(OUTPUT_DIR, f"{split_name}_pairs.npy")
    labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.npy")

    # Validate proper split, pairs, labels, and whether pairs and labels match
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing file: {pairs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    pairs = np.load(pairs_path)
    labels = np.load(labels_path).astype(int)

    # Validate proper split, pairs, labels, and whether pairs and labels match
    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)

    # Load all LFW images
    print("Loading images...")
    images = load_lfw_images()

    # Run evaluation using the selected validation threshold
    results = evaluate_pairs(
        images=images,
        pairs=pairs,
        labels=labels,
        threshold=selected_threshold,
        score_type=SCORE_FUNCTION,
    )

    # Validate the output structure of the metrics
    validate_metrics(results["metrics"])

    # Save the tracked run for experiment tracking
    save_run(
        run_id=f"sampled_test_eval_{ratio_tag}",
        split="test",
        data_version="test_pairs.npy",
        score_function=SCORE_FUNCTION,
        threshold_rule="maximize balanced_accuracy on sampled validation",
        selected_threshold=selected_threshold,
        metrics=results["metrics"],
        note=f"Final test evaluation using threshold from sampled validation ({ratio_tag})",
        extra={
            "ratio_tag": ratio_tag,
            "threshold_source": threshold_source,
            "sampled_data_version": sampled_pairs_filename(),
            "num_pairs": int(len(pairs))
        }
    )

    print(f"Split: {split_name}")
    print(f"Score type: {results['score_type']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Number of pairs: {results['num_pairs']}")
    print("Metrics:")
    for key, value in results["metrics"].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()