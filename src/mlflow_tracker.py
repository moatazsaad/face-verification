import os  
import json  
import subprocess  
import mlflow  


def _git_commit():
    
    try:       
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def init_mlflow(experiment_name="face-verification", tracking_uri="sqlite:///mlflow.db"):
    # Tell MLflow to use a SQLite database file for storing run metadata
    mlflow.set_tracking_uri(tracking_uri)

    # Create or select the experiment where runs will be grouped
    mlflow.set_experiment(experiment_name)

# Logs experiment parameters, metrics, and metadata to MLflow so the run appears in the MLflow UI
def log_run_to_mlflow(
    run_id,
    split,
    data_version,
    score_function,
    threshold_rule,
    selected_threshold,
    metrics,
    note,
    extra=None):
    
    # Start a new MLflow run with the given run name
    with mlflow.start_run(run_name=run_id):

        # Log descriptive tags 
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("split", split)
        mlflow.set_tag("data_version", data_version)
        mlflow.set_tag("score_function", score_function)
        mlflow.set_tag("threshold_rule", threshold_rule)
        mlflow.set_tag("code_version", _git_commit())
        mlflow.set_tag("note", note)

        # Log the selected threshold as a parameter
        mlflow.log_param("selected_threshold", round(float(selected_threshold), 4))

        # If extra metadata was provided
        if extra:
            # Keep track of whether any complex value exists
            has_complex = False

            # Loop through each extra metadata item
            for k, v in extra.items():
                # If the value is simple, log it directly as a parameter
                if isinstance(v, (str, int, float, bool)):
                    mlflow.log_param(k, v)
                else:
                    # Mark that at least one complex value exists
                    has_complex = True

            # If any complex metadata exists, save the full extra dict once as JSON
            if has_complex:
                path = "_extra.json"

                # Write the extra metadata to a temporary JSON file
                with open(path, "w") as f:
                    json.dump(extra, f, indent=2)

                # Upload the JSON file as an artifact for this run
                mlflow.log_artifact(path)

        # Loop through all evaluation metrics
        for k, v in metrics.items():
            # Only log numeric values as MLflow metrics
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

