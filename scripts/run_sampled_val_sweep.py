# Runs a threshold sweep on the sampled validation pairs
import os
import json
import numpy as np
from src.config import (
    OUTPUT_DIR, SCORE_FUNCTION, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
    VAL_NEGATIVE_RATIO, sampled_pairs_filename, sampled_labels_filename,
    sampled_sweep_filename, sampled_best_threshold_filename
)
from src.evaluation import load_lfw_images, compute_scores
from src.metrics import apply_threshold, compute_metrics
from src.run_tracker import save_run
from src.validation import (
    validate_split_name, validate_pairs, validate_labels,
    validate_pairs_and_labels_match, validate_scores
)


def main():
    split_name = "val_sampled"

    # Load the sampled validation pairs and labels
    pairs_path = os.path.join(OUTPUT_DIR, sampled_pairs_filename())
    labels_path = os.path.join(OUTPUT_DIR, sampled_labels_filename())

    # If files are missing
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing file: {pairs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    pairs = np.load(pairs_path)
    labels = np.load(labels_path).astype(int)
 
    # Validate proper split, pairs, labels, and whether pairs and labels match
    validate_split_name(split_name)
    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)

    # Load all LFW images
    print("Loading images...")
    images = load_lfw_images()

    print("Computing scores once...")
    scores = compute_scores(images, pairs, score_type=SCORE_FUNCTION)

    # Validate score count/shape after score computation
    validate_scores(scores, pairs)

    # Create threshold values for the sweep
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)

    sweep_results = []

    print("Sweeping thresholds...")
    for threshold in thresholds:
        # Convert scores into binary predictions
        preds = apply_threshold(scores, threshold, score_type=SCORE_FUNCTION)

        # Compute metrics at this threshold
        metrics = compute_metrics(labels, preds)

        # Compute false positive rate
        fpr = (
            metrics["fp"] / (metrics["fp"] + metrics["tn"])
            if (metrics["fp"] + metrics["tn"])
            else 0.0
        )

        # True positive rate equals recall
        tpr = metrics["recall"]

        # Save one row for this threshold
        row = {"threshold": float(threshold), "fpr": float(fpr), "tpr": float(tpr), **metrics,}
        sweep_results.append(row)

    # Get proper sweep name and name with best threshold from config
    sweep_name = sampled_sweep_filename()
    best_name = sampled_best_threshold_filename()

    # Save the full threshold sweep
    sweep_json_path = os.path.join(OUTPUT_DIR, sweep_name)
    with open(sweep_json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    # Select the best threshold using balanced accuracy
    best_result = max(sweep_results, key=lambda x: x["balanced_accuracy"])

    # Save the best threshold result
    best_json_path = os.path.join(OUTPUT_DIR, best_name)
    with open(best_json_path, "w") as f:
        json.dump(best_result, f, indent=2)        

    # Save tracked run
    save_run(
        run_id=f"sampled_val_sweep_neg{VAL_NEGATIVE_RATIO}x",
        split="val_sampled",
        data_version="val_pairs_sampled.npy",
        score_function=SCORE_FUNCTION,
        threshold_rule="maximize balanced_accuracy",
        selected_threshold=best_result["threshold"],
        metrics={
            "tp": best_result["tp"],
            "tn": best_result["tn"],
            "fp": best_result["fp"],
            "fn": best_result["fn"],
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "specificity": best_result["specificity"],
            "balanced_accuracy": best_result["balanced_accuracy"],
            "f1": best_result["f1"],
            "fpr": best_result["fpr"],
            "tpr": best_result["tpr"],
        },
        note="Post change validation threshold sweep using sampled validation negatives",
        extra={
            "negative_ratio": VAL_NEGATIVE_RATIO,
            "sweep_file": sweep_name,
            "best_file": best_name,
            "threshold_min": THRESHOLD_MIN,
            "threshold_max": THRESHOLD_MAX,
            "threshold_step": THRESHOLD_STEP,
            "num_pairs": int(len(pairs)),
            "num_thresholds": int(len(thresholds)),
        },
    )

    # Print results
    print("\nBest threshold based on balanced accuracy:")
    for k, v in best_result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()