# Evaluates the sampled validation dataset using the best threshold found during the sampled validation threshold sweep.
import os
import json
import numpy as np
from src.config import OUTPUT_DIR, SCORE_FUNCTION
from src.evaluation import load_lfw_images, evaluate_pairs
from src.run_tracker import save_run
from src.validation import validate_split_name, validate_pairs, validate_labels, validate_pairs_and_labels_match, validate_threshold, validate_metrics
from src.mlflow_tracker import log_run_to_mlflow
def main():

    # Load the best threshold selected during the sampled validation sweep
    best_path = os.path.join(OUTPUT_DIR, "val_sampled_best_threshold.json")

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing file: {best_path}")

    with open(best_path, "r") as f:
        best = json.load(f)

    selected_threshold = best["threshold"]
    validate_threshold(selected_threshold)

    split_name = "val_sampled"
    validate_split_name(split_name)

    # Load sampled validation pairs and labels
    pairs_path = os.path.join(OUTPUT_DIR, "val_pairs_sampled.npy")
    labels_path = os.path.join(OUTPUT_DIR, "val_labels_sampled.npy")

    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing file: {pairs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    pairs = np.load(pairs_path)
    labels = np.load(labels_path).astype(int)

    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)

    # Load all LFW images
    print("Loading images...")
    images = load_lfw_images()

    # Evaluate sampled validation data using the selected threshold
    results = evaluate_pairs(images=images, pairs=pairs, labels=labels, threshold=selected_threshold, score_type=SCORE_FUNCTION,)

    validate_metrics(results["metrics"])

    # Save this run so it appears in the tracked experiments
    save_run(
        run_id="sampled_val_eval_best",
        split="val_sampled",
        data_version="val_pairs_sampled.npy",
        score_function=SCORE_FUNCTION,
        threshold_rule="best threshold from sampled validation sweep",
        selected_threshold=selected_threshold,
        metrics=results["metrics"],
        note="Evaluation on sampled validation set using threshold selected from sweep",
        extra={
            "best_threshold_file": "val_sampled_best_threshold.json",
            "num_pairs": int(len(pairs)),
        }
    )
    
    # Alternative way; Log this evaluation run to MLflow for tracking
    log_run_to_mlflow(
        run_id="sampled_val_eval_best",
        split="val_sampled",
        data_version="val_pairs_sampled.npy",
        score_function=SCORE_FUNCTION,
        threshold_rule="best threshold from sampled validation sweep",
        selected_threshold=selected_threshold,
        metrics=results["metrics"],
        note="Evaluation on sampled validation set using threshold selected from sweep",
        extra={
            "best_threshold_file": "val_sampled_best_threshold.json",
            "num_pairs": int(len(pairs)),
        })
    # Print results to terminal
    print("Split: val_sampled")
    print(f"Score type: {results['score_type']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Number of pairs: {results['num_pairs']}")
    print("Metrics:")

    for key, value in results["metrics"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()