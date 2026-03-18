SEED = 47
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DATA_DIR = "data"
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

#  Milestone 2 settings 

# similarity score used by the verifier
SCORE_FUNCTION = "cosine"

# interpretation of score
SCORE_DIRECTION = "higher_is_more_similar"

# split used to choose threshold
THRESHOLD_SELECTION_SPLIT = "val"

# split used for final reporting
FINAL_EVAL_SPLIT = "test"

# threshold sweep range for validation experiments
THRESHOLD_MIN = -1.0
THRESHOLD_MAX = 1.0
THRESHOLD_STEP = 0.01

# negative sampling ratio for validation split
VAL_NEGATIVE_RATIO = 3     # keep 3 negatives for every positive pair

# directory for experiment runs
RUNS_DIR = "artifacts/runs"