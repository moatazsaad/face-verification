import numpy as np
import tensorflow_datasets as tfds
import json
import os
from src.data_ingest import ingest_lfw
from src.pair_gen import save_splits, generate_pairs

def main():
    print("Pair generation and saving in progress...")
    labels, dataset_split = ingest_lfw()
    save_splits(labels, dataset_split)

if __name__ == "__main__":
    main()