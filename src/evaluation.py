import numpy as np
import tensorflow_datasets as tfds
from src.config import DATA_DIR
from src.config import SCORE_FUNCTION
from src.similarity import cosine_similarity, euclidean_distance
from src.metrics import apply_threshold, compute_metrics

# Load all LFW images in the same deterministic sorted order used in data_ingest
def load_lfw_images():
    # Load LFW dataset from TFDS
    data = tfds.load("lfw:0.1.1", split="train", as_supervised=True, data_dir=DATA_DIR)

    entries = []

    # Store (label, original dataset index, image)
    for idx, (label, image) in enumerate(tfds.as_numpy(data)):
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        entries.append((label, idx, image))

    # Sorting first by label then by original index
    entries.sort(key=lambda x: (x[0], x[1]))

    # images in the same sorted order used for pair generation
    images = [entry[2] for entry in entries]

    return images

# Compute a similarity/distance score for each image pair
def compute_scores(images, pairs, score_type=SCORE_FUNCTION):
 
    scores = []

    # Loop through each pair of image indices
    for idx1, idx2 in pairs:
        img1 = images[idx1]
        img2 = images[idx2]

        if score_type == "cosine":
            _, score = cosine_similarity(img1, img2)
        elif score_type == "euclidean":
            _, score = euclidean_distance(img1, img2)
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")

        scores.append(score)

    return np.array(scores)

# Evaluate one split at one threshold, compute pair scores, convert scores to binary predictions, compute metrics
def evaluate_pairs(images, pairs, labels, threshold, score_type=SCORE_FUNCTION):

    labels = np.array(labels).astype(int)

    scores = compute_scores(images, pairs, score_type=score_type)
    predictions = apply_threshold(scores, threshold, score_type=score_type)
    metrics = compute_metrics(labels, predictions)

    return {
        "score_type": score_type,
        "threshold": threshold,
        "num_pairs": len(pairs),
        "scores": scores,
        "predictions": predictions,
        "metrics": metrics,
    }