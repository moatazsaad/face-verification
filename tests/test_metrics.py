<<<<<<< HEAD
from src.metrics import apply_threshold, confusion_counts, compute_metrics
import numpy as np
# NOT YET FINISHED
def test_apply_threshold_cosine():
    scores = np.array([0.9, 0.6, 0.2])
    preds = apply_threshold(scores, threshold=0.5, score_type="cosine")
    assert np.array_equal(preds, np.array([1, 1, 0]))

def test_apply_threshold_euclidean():
    scores = np.array([0.2, 0.5, 0.8])
    preds = apply_threshold(scores, threshold=0.5, score_type="euclidean")
    assert np.array_equal(preds, np.array([1, 1, 0]))

def test_confusion_counts():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    counts = confusion_counts(y_true, y_pred)
    assert counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}

def test_compute_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["specificity"] == 0.5
    assert m["balanced_accuracy"] == 0.5
    assert m["f1"] == 0.5
=======
import pytest
from src.metrics import compute_metrics

def test_compute_metrics_known_case():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
>>>>>>> origin/main
