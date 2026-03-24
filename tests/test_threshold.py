import numpy as np
from src.metrics import apply_threshold

def test_apply_threshold_cosine():
    scores = np.array([0.9, 0.7, 0.4, 0.2])
    threshold = 0.5

    preds = apply_threshold(scores, threshold, score_type="cosine")

    assert preds.tolist() == [1, 1, 0, 0]