import numpy as np
import os
from src.config import (
    OUTPUT_DIR,
    VAL_NEGATIVE_RATIO,
    SEED,
    sampled_pairs_filename,
    sampled_labels_filename,
)

def sample_validation_pairs():
    pairs = np.load(os.path.join(OUTPUT_DIR, "val_pairs.npy"))
    labels = np.load(os.path.join(OUTPUT_DIR, "val_labels.npy"))
    labels = np.array(labels)

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    num_pos = len(pos_indices)
    num_neg_keep = min(VAL_NEGATIVE_RATIO * num_pos, len(neg_indices))

    np.random.seed(SEED)
    sampled_neg_indices = np.random.choice(neg_indices, size=num_neg_keep, replace=False)

    selected_indices = np.concatenate([pos_indices, sampled_neg_indices])
    np.random.shuffle(selected_indices)

    sampled_pairs = pairs[selected_indices]
    sampled_labels = labels[selected_indices]

    pairs_name = sampled_pairs_filename()
    labels_name = sampled_labels_filename()

    np.save(os.path.join(OUTPUT_DIR, pairs_name), sampled_pairs)
    np.save(os.path.join(OUTPUT_DIR, labels_name), sampled_labels)

    print("Sampled validation pairs saved.")
    print(f"Ratio tag: neg{VAL_NEGATIVE_RATIO}x")
    print(f"Positives: {num_pos}")
    print(f"Negatives kept: {num_neg_keep}")
    print(f"Total pairs: {len(sampled_pairs)}")
    print(f"Saved: {pairs_name}, {labels_name}")

if __name__ == "__main__":
    sample_validation_pairs()