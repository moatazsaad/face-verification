#  no need for this
import os
import numpy as np

from src.config import OUTPUT_DIR, SCORE_FUNCTION
from src.evaluation import load_lfw_images, evaluate_pairs


# baseline threshold for a first test run
# later this will come from threshold sweep on validation
DEFAULT_THRESHOLD = 0.5


def main():
    split_name = "val"

    pairs_path = os.path.join(OUTPUT_DIR, f"{split_name}_pairs.npy")
    labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.npy")

    pairs = np.load(pairs_path)
    labels = np.load(labels_path)

    images = load_lfw_images()

    results = evaluate_pairs(
        images=images,
        pairs=pairs,
        labels=labels,
        threshold=DEFAULT_THRESHOLD,
        score_type=SCORE_FUNCTION,
    )

    print(f"Split: {split_name}")
    print(f"Score type: {results['score_type']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Number of pairs: {results['num_pairs']}")
    print("Metrics:")

    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()