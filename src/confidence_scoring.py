# Compute confidence metrics across thresholds on the sampled validation set, and select a best threshold based on balanced accuracy
import json
import os
import numpy as np
from src.config import (
    OUTPUT_DIR, SCORE_FUNCTION, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP
)

"""
Derive a confidence score from how far a similarity score is from the operating threshold.

This is a derived margin-based confidence, not a probabilistic calibration model. 
It expresses how far a score lies from the decision boundary.

Returns an ndarray the same size as scores of values between 0 and 1
"""
def compute_confidence_from_scores(scores, threshold, score_type=SCORE_FUNCTION, margin_scale=0.15):
    margins = np.abs(scores - threshold)
    confidence = np.clip(margins / margin_scale, 0.0, 1.0)
    return confidence.astype(np.float32)


# Compute confidence from single score
def compute_confidence_from_score(score, threshold, score_type=SCORE_FUNCTION, margin_scale=0.15):
    return float(compute_confidence_from_scores(scores=[score], threshold=threshold, score_type=score_type, margin_scale=margin_scale)[0])