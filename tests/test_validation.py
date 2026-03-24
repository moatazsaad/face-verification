import numpy as np
import pytest
from src.validation import validate_labels, validate_pairs

def test_validate_labels_binary_passes():
    labels = np.array([0, 1, 1, 0])
    validate_labels(labels)

def test_validate_labels_non_binary_fails():
    labels = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        validate_labels(labels)

def test_validate_pairs_shape_fails():
    pairs = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        validate_pairs(pairs)