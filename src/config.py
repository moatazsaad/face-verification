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


SCORE_FUNCTION = "cosine" # similarity score used by the verifier
SCORE_DIRECTION = "higher_is_more_similar" # interpretation of score
THRESHOLD_SELECTION_SPLIT = "val" # split used to choose threshold
FINAL_EVAL_SPLIT = "test" # split used for final reporting

# threshold sweep range for validation experiments
THRESHOLD_MIN = -1.0
THRESHOLD_MAX = 1.0
THRESHOLD_STEP = 0.01

# negative sampling ratio for validation split
VAL_NEGATIVE_RATIO = 3

# directory for experiment runs
RUNS_DIR = "artifacts/runs"

# MLflow settings
# Allows us to easily track and compare different runs, and log artifacts like the threshold sweep results
ENABLE_MLFLOW = True
MLFLOW_EXPERIMENT_NAME = "face-verification"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# The following functions save unique run names based on VAL_NEGATIVE_RATIO

def sampled_pairs_filename():
    return f"val_pairs_sampled_neg{VAL_NEGATIVE_RATIO}x.npy"

def sampled_labels_filename():
    return f"val_labels_sampled_neg{VAL_NEGATIVE_RATIO}x.npy"

def sampled_sweep_filename():
    return f"val_sampled_neg{VAL_NEGATIVE_RATIO}x_threshold_sweep.json"

def sampled_best_threshold_filename():
    return f"val_sampled_neg{VAL_NEGATIVE_RATIO}x_best_threshold.json"