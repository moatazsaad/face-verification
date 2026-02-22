import numpy as np
import time

# Python Loop Implementations

def cosine_similarity_loop(img1, img2):
    start = time.perf_counter()                     # Start timer to measure runtime

    vec1 = img1.astype(np.float64).ravel()             # Converts images to 1D float arrays for math operations, Flatten the arrays into 1D vectors do element wise operations
    vec2 = img2.astype(np.float64).ravel()

    dot_product  = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for i in range(len(vec1 )):
        dot_product += vec1 [i] * vec2[i]                          # dot product
        norm1 += vec1 [i] ** 2                          # sum of squares of arr1
        norm2 += vec2[i] ** 2

    norm1 = np.sqrt(norm1)
    norm2 = np.sqrt(norm2)

    if norm1 == 0 or norm2 == 0:                    # If either vector has zero length, cosine similarity is undefined. Return 0 safely.
        return time.perf_counter() - start, 0.0

    similarity = dot_product  / (norm1 * norm2)

    return time.perf_counter() - start, similarity


def euclidean_distance_loop(img1, img2):
    start = time.perf_counter()

    vec1  = img1.astype(np.float64).ravel()
    vec2 = img2.astype(np.float64).ravel()

    dist = 0.0
    for i in range(len(vec1 )):
        diff = vec1[i] - vec2[i]                      # Compute difference squared ((a[i]-b[i])^2)
        dist += diff * diff                     # Add to running total dist

    distance = np.sqrt(dist)                    # Take square root of the sum of squared differences

    return time.perf_counter() - start, distance


# Vectorized NumPy Implementations

def cosine_similarity(img1, img2):
    start = time.perf_counter()

    vec1 = img1.astype(np.float64).ravel()
    vec2 = img2.astype(np.float64).ravel()

    dot_product  = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return time.perf_counter() - start, 0.0

    similarity = dot_product  / (norm1 * norm2)

    return time.perf_counter() - start, similarity


def euclidean_distance(img1, img2):
    start = time.perf_counter()

    vec1 = img1.astype(np.float64).ravel()
    vec2 = img2.astype(np.float64).ravel()

    distance = np.linalg.norm(vec1 - vec2)

    return time.perf_counter() - start, distance


if __name__ == "__main__":
    # Create sample test images (simulating image data)
    np.random.seed(42)
    # Generate 150 random images of shape (100, 100, 3) to simulate real image data
    images = np.random.randint(0, 256, size=(150, 100, 100, 3), dtype=np.uint8)
    
    # 2 samples 
    img1 = images[10]
    img2 = images[100]

    # Cosine benchmark 
    loop_time_cos, loop_result_cos = cosine_similarity_loop(img1, img2)
    numpy_time_cos, numpy_result_cos = cosine_similarity(img1, img2)

    print("\nCosine Similarity")
    print(f"Loop Time: {loop_time_cos:.5f} sec")
    print(f"Numpy Time: {numpy_time_cos:.5f} sec")
    print(f"Speedup: {loop_time_cos / numpy_time_cos:.2f}x")
    assert np.isclose(loop_result_cos, numpy_result_cos), "Cosine results do not match!"
    print("Cosine correctness verified")


    # Euclidean benchmark
    loop_time_euc, loop_result_euc = euclidean_distance_loop(img1, img2)
    numpy_time_euc, numpy_result_euc = euclidean_distance(img1, img2)

    print("\nEuclidean Distance")
    print(f"Loop Time: {loop_time_euc:.5f} sec")
    print(f"Numpy Time: {numpy_time_euc:.5f} sec")
    print(f"Speedup: {loop_time_euc / numpy_time_euc:.2f}x")
    assert np.isclose(loop_result_euc, numpy_result_euc), "Euclidean results do not match!"
    print("Euclidean correctness verified")