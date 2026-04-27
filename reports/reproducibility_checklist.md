## Reproducibility Checklist

Follow the steps below to reproduce the core results of the face verification system.

### Clone Repository and Set Up Environment

```bash
git clone <repository-url>
cd face-verification
python3 -m venv tf_env
source tf_env/bin/activate
pip install -r requirements.txt
```

### Run CLI Inference (Single Pair)

```bash
python -m src.run_inference_cli --image1 examples/sample1.jpg --image2 examples/sample2.jpg
```

### Run Batch Inference

```bash
python -m src.run_inference_cli --folder examples/
```

### Run Tests

```bash
python -m pytest
```

### Locate Final Artifacts

* System Card: `reports/`
* Profiling Report: `reports/`
* Example inputs: `examples/`

### Final Release Version

```bash
git checkout v1.0-final
```
