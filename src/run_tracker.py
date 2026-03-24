import os
import json
import subprocess
from datetime import datetime
from src.config import RUNS_DIR

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def save_run(run_id, split, data_version, score_function, threshold_rule, selected_threshold, metrics, note, extra=None):
    os.makedirs(RUNS_DIR, exist_ok=True)

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
        "note": note,
    }

    if extra is not None:
        run_record["extra"] = extra

    save_path = os.path.join(RUNS_DIR, f"{run_id}.json")

    with open(save_path, "w") as f:
        json.dump(run_record, f, indent=2)

    print(f"Run saved to: {save_path}")