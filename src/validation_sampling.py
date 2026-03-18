import numpy as np
import os
from src.config import OUTPUT_DIR, VAL_NEGATIVE_RATIO, SEED


def sample_validation_pairs():

    # Load baseline validation pairs
    pairs = np.load(os.path.join(OUTPUT_DIR, "val_pairs.npy"))
    labels = np.load(os.path.join(OUTPUT_DIR, "val_labels.npy"))
    labels = np.array(labels)

    # Find positive and negative indices
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    # Count positives
    num_pos = len(pos_indices)

    # Decide how many negatives to keep
    num_neg_keep = VAL_NEGATIVE_RATIO * num_pos

    # Set deterministic random seed
    np.random.seed(SEED)

    # Sample negatives deterministically
    sampled_neg_indices = np.random.choice(neg_indices, size=num_neg_keep, replace=False)

    # Combine positives and sampled negatives
    selected_indices = np.concatenate([pos_indices, sampled_neg_indices])

    # Shuffle selected pairs
    np.random.shuffle(selected_indices)
    sampled_pairs = pairs[selected_indices]
    sampled_labels = labels[selected_indices]

    # Save new validation pair set
    np.save(os.path.join(OUTPUT_DIR, "val_pairs_sampled.npy"), sampled_pairs)
    np.save(os.path.join(OUTPUT_DIR, "val_labels_sampled.npy"), sampled_labels)

    print("Sampled validation pairs saved.")
    print(f"Positives: {num_pos}")
    print(f"Negatives: {num_neg_keep}")
    print(f"Total pairs: {len(sampled_pairs)}")
    
if __name__ == "__main__":
    sample_validation_pairs()