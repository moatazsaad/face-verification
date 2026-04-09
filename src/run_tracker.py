import os
import json
import subprocess
from datetime import datetime
from src.config import RUNS_DIR

# Utility functions for tracking and saving runs, including git commit retrieval and run record saving.
def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

# Save run details to a JSON file in the runs directory, including metadata and metrics.
def save_run(run_id, split, data_version, score_function, threshold_rule, selected_threshold, metrics, confidence_summary, note, extra=None):
    os.makedirs(RUNS_DIR, exist_ok=True)

    # Create a unique timestamp for the run

    run_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "code_version": get_git_commit(),
        "split": split,
        "data_version": data_version,
        "score_function": score_function,
        "threshold_rule": threshold_rule,
        "selected_threshold": selected_threshold,
        "metrics": metrics,
        "confidence_summary": confidence_summary,
        "note": note,
    }

    if extra is not None:
        run_record["extra"] = extra

    # Save path to run with run id and its corresponding timestamp
    save_path = os.path.join(RUNS_DIR, f"{run_id}.json")

    with open(save_path, "w") as f:
        json.dump(run_record, f, indent=2)

    print(f"Run saved to: {save_path}")