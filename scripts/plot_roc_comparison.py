import os
import json
import matplotlib.pyplot as plt
from src.config import OUTPUT_DIR, VAL_NEGATIVE_RATIO


def load_roc(path):
    with open(path, "r") as f:
        data = json.load(f)

    fpr = [row["fpr"] for row in data]
    tpr = [row["tpr"] for row in data]

    return fpr, tpr


def main():

    # Paths to sweep files
    val_path = os.path.join(OUTPUT_DIR, "val_threshold_sweep.json")
    sampled_path = os.path.join(OUTPUT_DIR, f"val_sampled_neg{VAL_NEGATIVE_RATIO}x_threshold_sweep.json")

    # Load ROC points
    val_fpr, val_tpr = load_roc(val_path)
    samp_fpr, samp_tpr = load_roc(sampled_path)

    plt.figure(figsize=(8, 6))

    # Plot full validation ROC
    plt.plot(
        val_fpr,
        val_tpr,
        label="Full Validation",
        linewidth=2
    )

    # Plot sampled validation ROC
    plt.plot(
        samp_fpr,
        samp_tpr,
        label="Sampled Validation",
        linewidth=2
    )

    # Random classifier reference
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # Save plot
    save_path = os.path.join(OUTPUT_DIR, "roc_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"ROC comparison plot saved to: {save_path}")


if __name__ == "__main__":
    main()