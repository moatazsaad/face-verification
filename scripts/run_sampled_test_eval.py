# Loads the best threshold found from the sampled validation sweep and uses it to evaluate the test set
import os
import numpy as np
import json
from src.config import OUTPUT_DIR, SCORE_FUNCTION
from src.evaluation import load_lfw_images, evaluate_pairs
from src.run_tracker import save_run
from src.validation import validate_split_name, validate_pairs, validate_labels, validate_pairs_and_labels_match, validate_threshold, validate_metrics

def main():

    # File containing the threshold selected from the validation sweep
    threshold_source = "val_sampled_best_threshold.json"
    best_path = os.path.join(OUTPUT_DIR, threshold_source)

    # Fail early if the threshold artifact is missing
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing file: {best_path}")

    # Prevent accidental data leakage:
    # the threshold shouldn't come from a test artifact
    if "test" in threshold_source.lower():
        raise ValueError("Threshold source must not come from test artifacts")

    # Load the selected threshold
    with open(best_path, "r") as f:
        best = json.load(f)

    SELECTED_THRESHOLD = best["threshold"]

    # Validate that the threshold is within an acceptable numeric range
    validate_threshold(SELECTED_THRESHOLD)

    # This script evaluates the held-out test split
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

    # Validate pair structure and labels
    validate_pairs(pairs)
    validate_labels(labels)

    # Ensure pairs and labels align in length
    validate_pairs_and_labels_match(pairs, labels)

    # Load all LFW images so pair indices can access them
    print("Loading images...")
    images = load_lfw_images()

    # Run evaluation using the selected validation threshold
    results = evaluate_pairs(
        images=images,
        pairs=pairs,
        labels=labels,
        threshold=SELECTED_THRESHOLD,
        score_type=SCORE_FUNCTION,
    )

    # Validate metrics output structure
    validate_metrics(results["metrics"])

    # Save the tracked run for experiment tracking
    save_run(
        run_id="sampled_test_eval",
        split="test",
        data_version="test_pairs.npy",
        score_function=SCORE_FUNCTION,
        threshold_rule="maximize balanced_accuracy on sampled validation",
        selected_threshold=SELECTED_THRESHOLD,
        metrics=results["metrics"],
        note="Final test evaluation using threshold selected from sampled validation sweep",
        extra={
            # Record where the threshold came from
            "threshold_source": threshold_source,
            "num_pairs": int(len(pairs))
        }
    )

    # Print results to terminal
    print(f"Split: {split_name}")
    print(f"Score type: {results['score_type']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Number of pairs: {results['num_pairs']}")
    print("Metrics:")

    for key, value in results["metrics"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()