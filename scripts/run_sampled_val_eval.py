# Evaluates the sampled validation dataset using the best threshold found during the sampled validation threshold sweep.
import os
import json
import numpy as np
from src.config import (
    OUTPUT_DIR, SCORE_FUNCTION, ENABLE_MLFLOW, VAL_NEGATIVE_RATIO,
    sampled_pairs_filename, sampled_labels_filename,
    sampled_best_threshold_filename
)
from src.evaluation import load_lfw_images, evaluate_pairs
from src.run_tracker import save_run
from src.validation import (
    validate_split_name, validate_pairs, validate_labels,
    validate_pairs_and_labels_match, validate_threshold, validate_metrics
)
from src.mlflow_tracker import init_mlflow, log_run_to_mlflow

def main():

    if ENABLE_MLFLOW:
        try:
            init_mlflow()           
        except Exception as e:
            print(f"Error initializing MLflow: {e}")

    threshold_source = sampled_best_threshold_filename()
    best_path = os.path.join(OUTPUT_DIR, threshold_source)

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing file: {best_path}")

    if "test" in threshold_source.lower():
        raise ValueError("Threshold source must not come from test artifacts")

    with open(best_path, "r") as f:
        best = json.load(f)

    # Get selected threshold
    selected_threshold = best["threshold"]
    validate_threshold(selected_threshold)

    # Sampled split
    split_name = "val_sampled"
    validate_split_name(split_name)

    # Load sampled validation pairs and labels
    split_name = "val_sampled"
    validate_split_name(split_name)

    # Get paths to pairs and labels
    pairs_path = os.path.join(OUTPUT_DIR, sampled_pairs_filename())
    labels_path = os.path.join(OUTPUT_DIR, sampled_labels_filename())

    # Check if the paths to the pairs and labels exist
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing file: {pairs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    # Load pairs and labels
    pairs = np.load(pairs_path)
    labels = np.load(labels_path).astype(int)

    # Validate proper split, pairs, labels, and whether pairs and labels match
    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)

    # Load all LFW images
    print("Loading images...")
    images = load_lfw_images()

    # Evaluate sampled validation data using the selected threshold
    results = evaluate_pairs(images=images, pairs=pairs, labels=labels, threshold=selected_threshold, score_type=SCORE_FUNCTION,)

    # Validate the metrics of the results
    validate_metrics(results["metrics"])

    # Save this run so it appears in the tracked experiments
    save_run(
        run_id=f"sampled_val_eval_neg{VAL_NEGATIVE_RATIO}x",
        split="val_sampled",
        data_version=sampled_pairs_filename(),
        score_function=SCORE_FUNCTION,
        threshold_rule="best threshold from sampled validation sweep",
        selected_threshold=selected_threshold,
        metrics=results["metrics"],
        note=f"Evaluation on sampled validation set using neg{VAL_NEGATIVE_RATIO}x",
        extra={
            "ratio_tag": f"neg{VAL_NEGATIVE_RATIO}x",
            "best_threshold_file": sampled_best_threshold_filename(),
            "num_pairs": int(len(pairs)),
        }
    )
    
    # Alternative way; Log this evaluation run to MLflow for tracking
    log_run_to_mlflow(
                run_id=f"sampled_val_eval_3{VAL_NEGATIVE_RATIO}x",
                split="val_sampled",
                data_version=sampled_pairs_filename(),
                score_function=SCORE_FUNCTION,
                threshold_rule="best threshold from sampled validation sweep",
                selected_threshold=selected_threshold,
                metrics=results["metrics"],
                note=f"Evaluation on sampled validation set using neg{VAL_NEGATIVE_RATIO}x",
                extra={
                    "ratio_tag": f"neg{VAL_NEGATIVE_RATIO}x",
                    "best_threshold_file": sampled_best_threshold_filename(),
                    "num_pairs": int(len(pairs)),
                }
            )
    # Print results to terminal
    print("Split: val_sampled")
    print(f"Score type: {results['score_type']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Number of pairs: {results['num_pairs']}")
    print("Metrics:")

    # Print results
    for key, value in results["metrics"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()