import os  
import json  
import numpy as np  
import matplotlib.pyplot as plt  
from src.config import OUTPUT_DIR, RUNS_DIR, SCORE_FUNCTION  
from src.evaluation import load_lfw_images, compute_scores  
from src.metrics import apply_threshold  

NUM_EXAMPLES = 3  # number of errors to save

def save_pair(img1, img2, title, save_path):
    plt.figure(figsize=(4,2))  

    plt.subplot(1,2,1)  
    plt.imshow(img1)  
    plt.axis("off")  

    plt.subplot(1,2,2)  
    plt.imshow(img2)  
    plt.axis("off")  

    plt.suptitle(title)  

    plt.tight_layout()  
    plt.savefig(save_path)  
    plt.close()  

def main():

    # load threshold from final test run
    run_path = os.path.join(RUNS_DIR, "sampled_test_eval.json")  # path to results

    with open(run_path) as f:
        run = json.load(f)  # load run metadata 

    threshold = run["selected_threshold"]  # extract threshold

    # load test data
    pairs = np.load(os.path.join(OUTPUT_DIR, "test_pairs.npy"))  # load test image pairs (indices)
    labels = np.load(os.path.join(OUTPUT_DIR, "test_labels.npy")).astype(int)  # load ground truth labels 

    print("Loading images...")
    images = load_lfw_images()  

    print("Computing scores...")
    scores = compute_scores(images, pairs, score_type=SCORE_FUNCTION)  # compute similarity score 

    preds = apply_threshold(scores, threshold, score_type=SCORE_FUNCTION)  # predictions 0/1

    # find error indices
    fp_idx = np.where((preds == 1) & (labels == 0))[0]  # FP
    fn_idx = np.where((preds == 0) & (labels == 1))[0]  # FN

    print("FP count:", len(fp_idx))  #  number of FP 
    print("FN count:", len(fn_idx))  #  number of FN 

    os.makedirs("outputs/error_examples", exist_ok=True)  

    # save FP 
    for i, idx in enumerate(fp_idx[:NUM_EXAMPLES]):  # loop over a few FP examples

        img1_idx, img2_idx = pairs[idx]  # get indices of the two images 

        save_pair(
            images[img1_idx],  
            images[img2_idx],  
            "False Positive",  # FP title 
            f"outputs/error_examples/fp_{i}.png"  # file path
        )

    # save FN examples
    for i, idx in enumerate(fn_idx[:NUM_EXAMPLES]):  # loop over a few FN examples

        img1_idx, img2_idx = pairs[idx]  # get indices of the two images in the pair

        save_pair(
            images[img1_idx],  # first image
            images[img2_idx],  # second image
            "False Negative",  # title for the saved image
            f"outputs/error_examples/fn_{i}.png"  # file path
        )

    print("Saved error examples to outputs/error_examples/")  

if __name__ == "__main__":
    main()  