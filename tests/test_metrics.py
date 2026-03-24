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
