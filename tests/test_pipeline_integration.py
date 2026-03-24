import json
import numpy as np
from pathlib import Path
from src.metrics import apply_threshold, compute_metrics
from src.run_tracker import save_run

def test_small_pipeline_integration(tmp_path, monkeypatch):
    # example
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    labels = np.array([1, 1, 0, 0])
    threshold = 0.5

    preds = apply_threshold(scores, threshold, score_type="cosine")
    metrics = compute_metrics(labels, preds)

    # Redirect run output to temporary folder
    monkeypatch.setattr("src.run_tracker.RUNS_DIR", str(tmp_path))

    save_run(
        run_id="test_run",
        split="val",
        data_version="tiny_fixture",
        score_function="cosine",
        threshold_rule="fixed",
        selected_threshold=threshold,
        metrics=metrics,
        note="integration test",
    )

    out_file = tmp_path / "test_run.json"
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert data["run_id"] == "test_run"
    assert data["split"] == "val"
    assert data["metrics"]["tp"] == 2
    assert data["metrics"]["tn"] == 2