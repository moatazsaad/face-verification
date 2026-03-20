# Runs a threshold sweep on the entire validation set
import os
import json
import numpy as np
from src.config import OUTPUT_DIR, SCORE_FUNCTION, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP
from src.evaluation import load_lfw_images, compute_scores
from src.metrics import apply_threshold, compute_metrics
from src.run_tracker import save_run
from src.validation import validate_split_name, validate_pairs, validate_labels, validate_pairs_and_labels_match, validate_scores
from src.mlflow_tracker import init_mlflow, log_run_to_mlflow

def main():
    
    init_mlflow()
    split_name = "val"

    pairs_path = os.path.join(OUTPUT_DIR, f"{split_name}_pairs.npy")
    labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.npy")

    # If files are missing, give error message
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing file: {pairs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")
    
    # Load pairs and labels
    pairs = np.load(pairs_path)
    labels = np.load(labels_path).astype(int)
    
    validate_split_name(split_name)
    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)
    
    print("Loading images...")
    images = load_lfw_images()

    print("Computing scores once...")
    scores = compute_scores(images, pairs, score_type=SCORE_FUNCTION)
    
    # Validate score count/shape after score computation
    validate_scores(scores, pairs)
                    
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

        row = {"threshold": float(threshold), "fpr": float(fpr), "tpr": float(tpr), **metrics,}
        sweep_results.append(row)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save full sweep artifact
    sweep_json_path = os.path.join(OUTPUT_DIR, f"{split_name}_threshold_sweep.json")
    with open(sweep_json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    # Select best threshold by balanced accuracy
    best_result = max(sweep_results, key=lambda x: x["balanced_accuracy"])

    # Save best-threshold artifact
    best_json_path = os.path.join(OUTPUT_DIR, f"{split_name}_best_threshold.json")
    with open(best_json_path, "w") as f:
        json.dump(best_result, f, indent=2)

    # Save tracked run
    save_run(
        run_id="baseline_val_sweep",
        split="val",
        data_version="val_pairs.npy",
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
        note="Baseline validation threshold sweep on full validation pairs",
        extra={
            "sweep_file": f"{split_name}_threshold_sweep.json",
            "best_file": f"{split_name}_best_threshold.json",
            "threshold_min": THRESHOLD_MIN,
            "threshold_max": THRESHOLD_MAX,
            "threshold_step": THRESHOLD_STEP,
            "num_pairs": int(len(pairs)),
            "num_thresholds": int(len(thresholds))
        }
    )

    print("\nBest threshold based on balanced accuracy:")
    for k, v in best_result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()