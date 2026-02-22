import os
import numpy as np
from src.data_ingest import ingest_lfw
from src.pair_gen import generate_pairs
from src.benchmark import benchmark
from src.config import OUTPUT_DIR


def main():
    labels, dataset_split = ingest_lfw()

    train_pairs, train_labels = generate_pairs(labels, dataset_split["train"])
    val_pairs, val_labels = generate_pairs(labels, dataset_split["val"])
    test_pairs, test_labels = generate_pairs(labels, dataset_split["test"])

    # Save all pairs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "train_pairs.npy"), train_pairs)
    np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), train_labels)
    np.save(os.path.join(OUTPUT_DIR, "val_pairs.npy"), val_pairs)
    np.save(os.path.join(OUTPUT_DIR, "val_labels.npy"), val_labels)
    np.save(os.path.join(OUTPUT_DIR, "test_pairs.npy"), test_pairs)
    np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), test_labels)
    print("All pairs saved successfully.")
    benchmark()

if __name__ == "__main__":
    main()