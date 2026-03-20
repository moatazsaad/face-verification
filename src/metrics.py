import numpy as np

# Convert scores into binary predictions
def apply_threshold(scores, threshold, score_type):
    scores = np.array(scores)

    if score_type == "cosine":
        # Higher score = more similar, return 1 if >= threshold otherwise 0
        return (scores >= threshold).astype(int)

    elif score_type == "euclidean":
        # Lower distance = more similar, return 1 if <= threshold otherwise 0
        return (scores <= threshold).astype(int)

    else:
        raise ValueError(f"Unsupported score_type: {score_type}")


# Compute TP, TN, FP, FN
def confusion_counts(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


# Compute evaluation metrics from predictions
def compute_metrics(y_true, y_pred):
    counts = confusion_counts(y_true, y_pred)

    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0                                          # Accuracy 
    precision = tp / (tp + fp) if (tp + fp) else 0.0                                        # Precision
    recall = tp / (tp + fn) if (tp + fn) else 0.0                                           # Recall 
    specificity = tn / (tn + fp) if (tn + fp) else 0.0                                      # Specificity
    balanced_accuracy = (recall + specificity) / 2 if total else 0.0                        # Balanced Accuracy
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0   # F1 Score

    return {
        **counts,                                   # Include unpacked metrics from confusion_counts
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }