import os
import json
import matplotlib.pyplot as plt
import numpy as np

from src.config import RUNS_DIR


def main():
    # Load the final tracked test run
    run_path = os.path.join(RUNS_DIR, "sampled_test_eval.json")

    with open(run_path, "r") as f:
        run_data = json.load(f)

    # Extract confusion matrix values
    tp = run_data["metrics"]["tp"]
    tn = run_data["metrics"]["tn"]
    fp = run_data["metrics"]["fp"]
    fn = run_data["metrics"]["fn"]

    # Arrange as a standard confusion matrix:
    # rows = actual class, columns = predicted class
    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()

    # Axis labels
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix: Test Set Evaluation")

    # Write values inside the cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                ha="center",
                va="center"
            )

    plt.tight_layout()

    # Save the plot in the same folder as the tracked run
    save_path = os.path.join(RUNS_DIR, "test_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Confusion matrix saved to: {save_path}")


if __name__ == "__main__":
    main()
    # python -m scripts.plot_confusion_matrix