import numpy as np
import pytest

from src.validation import (
    validate_split_name,
    validate_pairs,
    validate_labels,
    validate_pairs_and_labels_match,
    validate_threshold,
    validate_scores,
    validate_metrics,
)

def test_validate_split_name_valid():
    validate_split_name("train")
    validate_split_name("val")
    validate_split_name("test")
    validate_split_name("val_sampled")

def test_validate_pairs_valid():
    pairs = np.array([[0, 1], [2, 3]])
    validate_pairs(pairs)

def test_validate_pairs_wrong_shape():
    with pytest.raises(ValueError):
        validate_pairs(np.array([0, 1, 2]))

def test_validate_pairs_empty():
    with pytest.raises(ValueError):
        validate_pairs(np.empty((0, 2), dtype=int))

def test_validate_labels_valid():
    labels = np.array([0, 1, 1, 0])
    validate_labels(labels)

def test_validate_labels_non_binary():
    with pytest.raises(ValueError):
        validate_labels(np.array([0, 1, 2]))

def test_validate_labels_not_1d():
    with pytest.raises(ValueError):
        validate_labels(np.array([[0, 1], [1, 0]]))

def test_validate_pairs_and_labels_match_valid():
    pairs = np.array([[0, 1], [2, 3]])
    labels = np.array([1, 0])
    validate_pairs_and_labels_match(pairs, labels)

def test_validate_pairs_and_labels_match_invalid():
    pairs = np.array([[0, 1], [2, 3]])
    labels = np.array([1])
    with pytest.raises(ValueError):
        validate_pairs_and_labels_match(pairs, labels)

def test_validate_threshold_valid():
    validate_threshold(0.5)

def test_validate_threshold_invalid_type():
    with pytest.raises(TypeError):
        validate_threshold("0.5")

def test_validate_threshold_out_of_range():
    with pytest.raises(ValueError):
        validate_threshold(2.0)

def test_validate_scores_valid():
    pairs = np.array([[0, 1], [2, 3]])
    scores = np.array([0.8, 0.1])
    validate_scores(scores, pairs)

def test_validate_scores_wrong_length():
    pairs = np.array([[0, 1], [2, 3]])
    scores = np.array([0.8])
    with pytest.raises(ValueError):
        validate_scores(scores, pairs)

def test_validate_metrics_valid():
    metrics = {
        "tp": 1, "tn": 1, "fp": 0, "fn": 0,
        "accuracy": 1.0, "precision": 1.0, "recall": 1.0,
        "specificity": 1.0, "balanced_accuracy": 1.0, "f1": 1.0
    }
    validate_metrics(metrics)

def test_validate_metrics_missing_key():
    metrics = {
        "tp": 1, "tn": 1, "fp": 0, "fn": 0,
        "accuracy": 1.0, "precision": 1.0, "recall": 1.0,
        "specificity": 1.0, "balanced_accuracy": 1.0
    }
    with pytest.raises(ValueError):
        validate_metrics(metrics)
