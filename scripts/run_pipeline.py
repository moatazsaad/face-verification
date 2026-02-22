import numpy as np
from src.data_ingest import ingest_lfw
from src.pair_gen import save_splits
from src.benchmark import benchmark
from src.config import OUTPUT_DIR

def main():
    labels, dataset_split = ingest_lfw()

    # Generate all pairs and save automatically
    save_splits(labels, dataset_split)

    benchmark()

if __name__ == "__main__":
    main()