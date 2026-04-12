import subprocess
import sys


# Test threshold logic 
def test_threshold():
    score = 0.3
    threshold = 0.29
    decision = int(score >= threshold)
    assert decision == 1


# Test CLI runs successfully 
def test_cli_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.run_inference_cli",
            "--image1",
            "examples/sample1.jpg",
            "--image2",
            "examples/sample2.jpg",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "score" in result.stdout.lower()
    assert "threshold" in result.stdout.lower()
    assert "confidence" in result.stdout.lower()
    assert "latency" in result.stdout.lower()
    
    