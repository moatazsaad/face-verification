SEED = 47
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

OUTPUT_DIR = "artifacts"


PAIR_POLICY_CONFIG = {
    "train": {
        "num_positive_pairs": "all possible",                           # We use all possible positive pairs for this split
        "num_negative_pairs": "all possible",                           # We use all possible negative pairs for this split
        "sampling_method": "exhaustive, shuffled after combining"       # How pairs are generated
    },
    "val": {                                                            # Same policy for validation split
        "num_positive_pairs": "all possible",
        "num_negative_pairs": "all possible",
        "sampling_method": "exhaustive, shuffled after combining"
    },
    "test": {
        "num_positive_pairs": "all possible",
        "num_negative_pairs": "all possible",
        "sampling_method": "exhaustive, shuffled after combining"
    }
}