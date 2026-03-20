from src.evaluation import compute_scores
import numpy as np
## NOT YET FINISHED
def test_compute_scores_cosine():
    img_a = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_b = np.array([[1, 0], [0, 0]], dtype=np.float32)
    img_c = np.array([[0, 1], [0, 0]], dtype=np.float32)

    images = [img_a, img_b, img_c]
    pairs = np.array([[0, 1], [0, 2]])

    scores = compute_scores(images, pairs, score_type="cosine")

    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)