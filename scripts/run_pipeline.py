from src.data_ingest import ingest_lfw
from src.pair_gen import generate_pairs
from src.benchmark import benchmark

def main():
    labels, dataset_split = ingest_lfw()

    train_pairs, train_labels = generate_pairs(labels, dataset_split["train"])
    val_pairs, val_labels = generate_pairs(labels, dataset_split["val"])
    test_pairs, test_labels = generate_pairs(labels, dataset_split["test"])
    benchmark()

if __name__ == "__main__":
    main()
