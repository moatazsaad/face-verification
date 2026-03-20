from src.evaluation import evaluate_pairs
import numpy as np
# NOT YET FINISHED
def test_evaluate_pairs_cosine():
    img_a = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_b = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_c = np.array([[0, 1], [0, 0]], dtype=np.float32)

    images = [img_a, img_b, img_c]
    pairs = np.array([[0, 1], [0, 2]])
    labels = np.array([1, 0])

    result = evaluate_pairs(images, pairs, labels, threshold=0.5, score_type="cosine")

    assert np.array_equal(result["predictions"], np.array([1, 0]))
    assert result["metrics"]["tp"] == 1
    assert result["metrics"]["tn"] == 1
    assert result["metrics"]["fp"] == 0
    assert result["metrics"]["fn"] == 0
    assert result["metrics"]["accuracy"] == 1.0