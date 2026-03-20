# Runs validation evaluation on the full validation set using the threshold selected from the validation sweep
import os
import json
import numpy as np
from src.config import OUTPUT_DIR, SCORE_FUNCTION
from src.evaluation import load_lfw_images, evaluate_pairs
from src.run_tracker import save_run
from src.validation import validate_split_name, validate_pairs, validate_labels, validate_pairs_and_labels_match, validate_threshold, validate_metrics

def main():

    # We are evaluating the validation split
    split_name = "val"
    validate_split_name(split_name)

    # Load the best threshold selected during the validation sweep
    best_path = os.path.join(OUTPUT_DIR, "val_best_threshold.json")

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing file: {best_path}")

    with open(best_path, "r") as f:
        best = json.load(f)

    selected_threshold = best["threshold"]
    validate_threshold(selected_threshold)

    # Load validation pairs and labels (full validation set)
    pairs_path = os.path.join(OUTPUT_DIR, f"{split_name}_pairs.npy")
    labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.npy")

    # Check that files exist before attempting to load them
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing file: {pairs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    pairs = np.load(pairs_path)
    labels = np.load(labels_path).astype(int)

    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)

    # Load all LFW images so pair indices can access them
    print("Loading images...")
    images = load_lfw_images()

    # Run evaluation using the selected threshold
    results = evaluate_pairs(
        images=images,
        pairs=pairs,
        labels=labels,
        threshold=selected_threshold,
        score_type=SCORE_FUNCTION,
    )

    validate_metrics(results["metrics"])

    # Save this experiment as a tracked run
    save_run(
        run_id="baseline_val_selected",
        split="val",
        data_version="val_pairs.npy",
        score_function=SCORE_FUNCTION,
        threshold_rule="maximize balanced_accuracy on validation sweep",
        selected_threshold=selected_threshold,
        metrics=results["metrics"],
        note="Validation evaluation using threshold selected from validation sweep",
        extra={
            "num_pairs": int(len(pairs)),
            "threshold_source": "val_best_threshold.json"
        }
    )

    # Print results to terminal
    print(f"Split: {split_name}")
    print(f"Score type: {results['score_type']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Number of pairs: {results['num_pairs']}")
    print("Metrics:")

    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()