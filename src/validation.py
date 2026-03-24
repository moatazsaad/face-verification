import numpy as np

# Allowed dataset split names 
VALID_SPLITS = {"train", "val", "test", "val_sampled"}


def validate_split_name(split_name):
    # This prevents unexpected split names
    if split_name not in VALID_SPLITS:
        raise ValueError(f"Invalid split name: {split_name}")


def validate_pairs(pairs):
    # Pairs should be numpy array 
    if not isinstance(pairs, np.ndarray):
        raise TypeError("pairs must be a numpy array")

    # Pairs should be 2D with exactly two columns (each row = one pair)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (N, 2)")

    # Ensure the array contains at least one pair
    if len(pairs) == 0:
        raise ValueError("pairs array is empty")

def validate_labels(labels):
    # Labels should be a numpy array.
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy array")

    # labels should be a single dimension vector
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")

    # labels cannot be empty
    if len(labels) == 0:
        raise ValueError("labels array is empty")

    # Ensure labels are only binary values
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")


def validate_pairs_and_labels_match(pairs, labels):
    # Number of labels matches the number of pairs
    if len(pairs) != len(labels):
        raise ValueError("number of pairs must match number of labels")


def validate_threshold(threshold, min_value=-1.0, max_value=1.0):
    # Threshold must be numeric
    if not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError("threshold must be numeric")

    if threshold < min_value or threshold > max_value:
        raise ValueError(f"threshold {threshold} is outside allowed range [{min_value}, {max_value}]")


def validate_scores(scores, pairs):
    # Scores must be a numpy array
    if not isinstance(scores, np.ndarray):
        raise TypeError("scores must be a numpy array")

    # scores should be a vector
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array")

    # Ensure each pair has a corresponding score
    if len(scores) != len(pairs):
        raise ValueError("number of scores must match number of pairs")


def validate_metrics(metrics):
    # Validates all required evaluation metrics exist in the metrics dictionary
    required = {"tp", "tn", "fp", "fn", "accuracy", "precision", "recall","specificity", "balanced_accuracy", "f1"}

    # Identify any missing metrics
    missing = required - set(metrics.keys())

    if missing:
        raise ValueError(f"metrics missing required fields: {sorted(missing)}")