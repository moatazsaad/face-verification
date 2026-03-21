import numpy as np

from src.validation import (
    validate_pairs,
    validate_labels,
    validate_pairs_and_labels_match,
    validate_metrics,
)
from src.evaluation import evaluate_pairs

def test_small_pipeline():
    img_a = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_b = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_c = np.array([[0, 1], [0, 0]], dtype=np.float32)

    images = [img_a, img_b, img_c]
    pairs = np.array([[0, 1], [0, 2]])
    labels = np.array([1, 0])

    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)

    result = evaluate_pairs(
        images=images,
        pairs=pairs,
        labels=labels,
        threshold=0.5,
        score_type="cosine"
    )

    assert "scores" in result
    assert "predictions" in result
    assert "metrics" in result
    assert len(result["scores"]) == len(labels)
    assert len(result["predictions"]) == len(labels)

    validate_metrics(result["metrics"])
    assert result["metrics"]["accuracy"] == 1.0