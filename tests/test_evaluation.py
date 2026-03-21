import numpy as np

from src.evaluation import compute_scores, evaluate_pairs

def test_compute_scores_cosine():
    img_a = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_b = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_c = np.array([[0, 1], [0, 0]], dtype=np.float32)

    images = [img_a, img_b, img_c]
    pairs = np.array([[0, 1], [0, 2]])

    scores = compute_scores(images, pairs, score_type="cosine")

    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)

def test_compute_scores_euclidean():
    img_a = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_b = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_c = np.array([[0, 1], [0, 0]], dtype=np.float32)

    images = [img_a, img_b, img_c]
    pairs = np.array([[0, 1], [0, 2]])

    scores = compute_scores(images, pairs, score_type="euclidean")

    assert np.isclose(scores[0], 0.0)
    assert scores[1] > 0.0

def test_evaluate_pairs_cosine():
    img_a = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_b = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_c = np.array([[0, 1], [0, 0]], dtype=np.float32)

    images = [img_a, img_b, img_c]
    pairs = np.array([[0, 1], [0, 2]])
    labels = np.array([1, 0])

    result = evaluate_pairs(
        images=images,
        pairs=pairs,
        labels=labels,
        threshold=0.5,
        score_type="cosine"
    )

    assert np.array_equal(result["predictions"], np.array([1, 0]))
    assert result["metrics"]["tp"] == 1
    assert result["metrics"]["tn"] == 1
    assert result["metrics"]["fp"] == 0
    assert result["metrics"]["fn"] == 0
    assert result["metrics"]["accuracy"] == 1.0