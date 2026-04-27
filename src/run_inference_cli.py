import os
from itertools import combinations
import argparse
import time
import numpy as np
from PIL import Image
from src.config import SCORE_FUNCTION, OPERATING_THRESHOLD
from src.confidence_scoring import compute_confidence_from_scores
from src.metrics import apply_threshold
from src.similarity import cosine_similarity, euclidean_distance
import warnings

# Suppress warnings for clean CLI outputs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  
    
# Load image and convert it to RGB
def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

#  Measure latency in milliseconds
def compute_latency_ms(time):
    return time * 1000



# Function to compute embedding for a single pair
def run_single_inference(img1, img2, pair_id=None):

    if pair_id is None:
        pair_id = f"({img1}, {img2})"

    # JPG image arr
    img1 = load_image(img1)
    img2 = load_image(img2)

    preprocessing_time = 0.0
    embedding_time = 0.0
    score_time = 0.0

    if SCORE_FUNCTION in ["cosine_with_embeddings", "euclidean_with_embeddings"]:
        from src.generate_image_embeddings import (
            preprocess_image,
            extract_embedding_from_preprocessed,
            compute_scores_from_embeddings,
        )

        start = time.perf_counter()
        img1_preprocessed = preprocess_image(img1)
        img2_preprocessed = preprocess_image(img2)
        preprocessing_time = time.perf_counter() - start

        start = time.perf_counter()
        emb1 = extract_embedding_from_preprocessed(img1_preprocessed)
        emb2 = extract_embedding_from_preprocessed(img2_preprocessed)
        embedding_time = time.perf_counter() - start

        start = time.perf_counter()
        scores = compute_scores_from_embeddings([emb1, emb2], pairs=[(0, 1)])
        score = float(scores[0])
        score_time = time.perf_counter() - start

    else:
        start = time.perf_counter()
        if SCORE_FUNCTION == "cosine":
            _, score = cosine_similarity(img1, img2)
        elif SCORE_FUNCTION == "euclidean":
            _, score = euclidean_distance(img1, img2)
        else:
            raise ValueError(f"Unsupported SCORE_FUNCTION: {SCORE_FUNCTION}")
        score_time = time.perf_counter() - start
        scores = np.array([score], dtype=np.float32)
        score = float(scores[0])

    decision = int(apply_threshold(scores, OPERATING_THRESHOLD, score_type=SCORE_FUNCTION)[0])
    decision_str = "match" if decision == 1 else "non-match"
    confidence = float(
        compute_confidence_from_scores(
            scores=scores,
            threshold=OPERATING_THRESHOLD,
            score_type=SCORE_FUNCTION,
        )[0]
    )

    total_time = preprocessing_time + embedding_time + score_time

    result = {
        "pair_id": pair_id,
        "score": score,
        "score_type": SCORE_FUNCTION,
        "threshold": OPERATING_THRESHOLD,
        "decision": decision_str,
        "confidence": confidence,
        "preprocessing_time_ms": compute_latency_ms(preprocessing_time),
        "embedding_time_ms": compute_latency_ms(embedding_time),
        "score_time_ms": compute_latency_ms(score_time),
        "total_time_ms": compute_latency_ms(total_time),
    }
    return result

# Function to print structured inferences for single inference pair
def print_single_inference_result(result):
    print("\nInference Result")
    print("------------------")
    
    for k, v in result.items():
        if k.endswith("_time_ms"):
            print(f"{k}={v:.2f}")
        else:
            print(f"{k}={v}")




# Run batch inference

# Function used to generate pairs and pair_ids for the batch inference
def generate_all_pairs_from_folder(folder_path):
    image_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    pairs = list(combinations(image_paths, 2))
    pair_ids = [
        f"{os.path.basename(p1)} vs {os.path.basename(p2)}"
        for p1, p2 in pairs
    ]

    return pairs, pair_ids


# Function to run batch inference:
# - loads each unique image once
# - preprocesses all unique images
# - extracts embeddings once per unique image
# - scores all pairs using vectorized NumPy operations
def run_vectorized_batch_inference(pairs, pair_ids=None):

    if pair_ids is None:
        pair_ids = [f"{p1} vs {p2}" for p1, p2 in pairs]

    if SCORE_FUNCTION not in ["cosine_with_embeddings", "euclidean_with_embeddings"]:
        raise ValueError(
            "Vectorized batch inference currently supports embedding-based scoring only"
        )

    from src.generate_image_embeddings import (
        preprocess_image,
        extract_embedding_from_preprocessed,
    )

    # Collect unique image paths
    unique_paths = sorted(set([p for pair in pairs for p in pair]))
    path_to_idx = {path: i for i, path in enumerate(unique_paths)}

    # Load and preprocess images
    start = time.perf_counter()
    loaded_images = [load_image(path) for path in unique_paths]
    preprocessed_images = [preprocess_image(img) for img in loaded_images]
    preprocessing_time = time.perf_counter() - start

    # Extract embeddings once per unique image
    start = time.perf_counter()
    embeddings = [extract_embedding_from_preprocessed(img) for img in preprocessed_images]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embedding_time = time.perf_counter() - start

    # Convert pair paths to embedding indices
    pair_indices = np.array([(path_to_idx[p1], path_to_idx[p2]) for p1, p2 in pairs], dtype=np.int64)
    emb1 = embeddings[pair_indices[:, 0]]
    emb2 = embeddings[pair_indices[:, 1]]

    # Vectorized scoring
    start = time.perf_counter()

    if SCORE_FUNCTION == "cosine_with_embeddings":
        scores = np.sum(emb1 * emb2, axis=1)
    elif SCORE_FUNCTION == "euclidean_with_embeddings":
        scores = np.linalg.norm(emb1 - emb2, axis=1)
    else:
        raise ValueError(f"Unsupported SCORE_FUNCTION: {SCORE_FUNCTION}")

    score_time = time.perf_counter() - start

    # Vectorized decisions and confidence
    decisions = apply_threshold(
        scores,
        OPERATING_THRESHOLD,
        score_type=SCORE_FUNCTION,
    )

    confidences = compute_confidence_from_scores(
        scores=scores,
        threshold=OPERATING_THRESHOLD,
        score_type=SCORE_FUNCTION,
    )

    total_time = preprocessing_time + embedding_time + score_time

    # Format results
    results = []

    for i, pair_id in enumerate(pair_ids):
        decision_str = "match" if int(decisions[i]) == 1 else "non-match"

        results.append({
            "pair_id": pair_id,
            "score": float(scores[i]),
            "score_type": SCORE_FUNCTION,
            "threshold": OPERATING_THRESHOLD,
            "decision": decision_str,
            "confidence": float(confidences[i]),
            "preprocessing_time_ms": compute_latency_ms(preprocessing_time),
            "embedding_time_ms": compute_latency_ms(embedding_time),
            "score_time_ms": compute_latency_ms(score_time),
            "total_time_ms": compute_latency_ms(total_time),
        })

    return results



# Main method to parce CLI arguments and run either single inference or batch inference based on the provided arguments
def main():
    parser = argparse.ArgumentParser(description="Face verification CLI")

    # For single pair inference
    parser.add_argument("--image1", help="Path to first image")
    parser.add_argument("--image2", help="Path to second image")
    # For batch inference
    parser.add_argument("--folder", help="Folder containing images for batch inference")

    args = parser.parse_args()

    if args.image1 and args.image2:
        result = run_single_inference(args.image1, args.image2)
        print_single_inference_result(result)
    elif args.folder: # Run batch inference if folder is provided
        pairs, pair_ids = generate_all_pairs_from_folder(args.folder)

        results = run_vectorized_batch_inference(
            pairs=pairs,
            pair_ids=pair_ids,
        )

        print("\nBatch Inference Results")
        print("=======================")

        for r in results:
            print(
                f"{r['pair_id']} -> "
                f"{r['decision']} "
                f"(score={r['score']:.4f}, confidence={r['confidence']:.4f})"
            )

        print("\nBatch Timing Summary")
        print("====================")
        print(f"num_pairs={len(results)}")
        print(f"preprocessing_time_ms={results[0]['preprocessing_time_ms']:.2f}")
        print(f"embedding_time_ms={results[0]['embedding_time_ms']:.2f}")
        print(f"score_time_ms={results[0]['score_time_ms']:.2f}")
        print(f"total_time_ms={results[0]['total_time_ms']:.2f}")

    else:
        raise ValueError("Provide either --image1 and --image2, or --folder")



# python -m src.run_inference_cli --image1 "examples/sample1.jpg" --image2 "examples/sample2.jpg"
# to run inside docker container:
# docker run --rm face-verification --image1 "/app/examples/sample1.jpg" --image2 "/app/examples/sample2.jpg"

# Example batch inference command:
# python -m src.run_inference_cli --folder examples/
if __name__ == "__main__":
    main()