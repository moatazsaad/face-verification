import argparse
import time
import numpy as np
from PIL import Image
from src.config import SCORE_FUNCTION, OPERATING_THRESHOLD, OUTPUT_DIR
from src.confidence_scoring import compute_confidence_from_scores
from src.evaluation import compute_scores
from src.metrics import apply_threshold
from src.similarity import cosine_similarity, euclidean_distance
import warnings

# Suppress warnings for clean CLI outputs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  
    
# Load image and convert it to RGB
def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

'''# Compute similarity score between two embeddings
def compute_pair_score(emb1, emb2):
    if emb1 is None or emb2 is None:
        return -1.0
    return float(np.dot(emb1, emb2))'''

'''# Apply threshold to convert score into binary decision
def apply_decision(score, threshold):
    return int(score >= threshold)'''

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
        "preprocessing_time_ms": f"{compute_latency_ms(preprocessing_time):.2f}",
        "embedding_time_ms": f"{compute_latency_ms(embedding_time):.2f}",
        "score_time_ms": f"{compute_latency_ms(score_time):.2f}",
        "total_time_ms": f"{compute_latency_ms(total_time):.2f}",
    }
    return result



# Function to print structured inferences for single inference pair
def print_single_inference_result(result):
    print("\nInference Result")
    print("------------------")
    print(*[f"{k}={v}" for k, v in result.items()], sep="\n")


def main():
    # User  provide 2 image paths
    parser = argparse.ArgumentParser(description="Run face verification on a pair of images")
    parser.add_argument("--image1", required=True, help="Path to first image")
    parser.add_argument("--image2", required=True, help="Path to second image")
    args = parser.parse_args()    

    # Print results
    result = run_single_inference(args.image1, args.image2, pair_id=None)
    print_single_inference_result(result)


if __name__ == "__main__":
    main()
    
# python -m src.run_inference_cli --image1 "examples/sample1.jpg" --image2 "examples/sample2.jpg"
# to run inside docker container:
# docker run --rm face-verification --image1 "/app/examples/sample1.jpg" --image2 "/app/examples/sample2.jpg"