import os
import numpy as np

from src.config import OUTPUT_DIR, SCORE_FUNCTION
from src.evaluation import load_lfw_images, compute_scores
from src.validation import (
    validate_split_name,
    validate_pairs,
    validate_labels,
    validate_pairs_and_labels_match,
    validate_scores,
)


def main():
    # Change to "test" if you want to inspect the held-out test split
    split_name = "val"
    validate_split_name(split_name)

    pairs_path = os.path.join(OUTPUT_DIR, f"{split_name}_pairs.npy")
    labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.npy")

    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing file: {pairs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    pairs = np.load(pairs_path)
    labels = np.load(labels_path).astype(int)

    validate_pairs(pairs)
    validate_labels(labels)
    validate_pairs_and_labels_match(pairs, labels)

    print("Loading images...")
    images = load_lfw_images()

    print("Computing scores...")
    scores = compute_scores(images, pairs, score_type=SCORE_FUNCTION)

    validate_scores(scores, pairs)

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0:
        raise ValueError("No positive pairs found.")
    if len(neg_scores) == 0:
        raise ValueError("No negative pairs found.")

    print("\nPositive-score summary:")
    print(f"  min : {pos_scores.min():.6f}")
    print(f"  mean: {pos_scores.mean():.6f}")
    print(f"  max : {pos_scores.max():.6f}")

    print("\nNegative-score summary:")
    print(f"  min : {neg_scores.min():.6f}")
    print(f"  mean: {neg_scores.mean():.6f}")
    print(f"  max : {neg_scores.max():.6f}")

    perfect_separation = float(neg_scores.max()) < float(pos_scores.min())

    print("\nSeparation check:")
    print(f"  max negative score: {neg_scores.max():.6f}")
    print(f"  min positive score: {pos_scores.min():.6f}")
    print(f"  perfect separation: {perfect_separation}")

    if not perfect_separation:
        overlap_count = np.sum(
            (scores >= pos_scores.min()) & (scores <= neg_scores.max())
        )
        print(f"  rough overlap-region count: {int(overlap_count)}")


if __name__ == "__main__":
    main()