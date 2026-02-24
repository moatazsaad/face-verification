import numpy as np
import tensorflow_datasets as tfds
import json
import os
from src.config import SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, OUTPUT_DIR

def ingest_lfw():
    
    np.random.seed(SEED)
    # Load dataset (default version 0.1.1)
    data, info = tfds.load("lfw:0.1.1", split="train", as_supervised=True, with_info=True)

    labels = []
    for label, _ in tfds.as_numpy(data):
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        labels.append(label)

    labels = np.array(labels)
    
    # Number of samples in each split
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train_count = int(len(labels) * TRAIN_RATIO)
    val_count = int(len(labels) * VAL_RATIO)
    
    # Indices for each split
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]

    # Create a splits dictionary
    dataset_split = {
    "train": train_indices,
    "val": val_indices,
    "test": test_indices
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    manifest = {
        "seed": SEED,
        "split_criteria": "80% train,10% val,10% test",
        "total_images": len(labels),
        "num_identities": len(set(labels)),
        "train_size": len(dataset_split["train"]),
        "val_size": len(dataset_split["val"]),
        "test_size": len(dataset_split["test"]),
        "data_source": {
            "tfds_name": info.name,
            "version": str(info.version)
        },
        "cache_directory": info.data_dir,
    }

    with open(os.path.join(OUTPUT_DIR, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent = 2)

    print(f"Dataset ingested successfully. Manifest saved to {OUTPUT_DIR}/dataset_manifest.json")
    return labels, dataset_split       

if __name__=="__main__":
    labels, dataset_split = ingest_lfw()
    print(f"Number of images: {len(labels)}\nTrain size: {len(dataset_split['train'])}\nVal size: {len(dataset_split['val'])}\nTest size:{len(dataset_split['test'])}")
