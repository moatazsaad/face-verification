import os

import tensorflow_datasets as tfds
import numpy as np
from src.similarity import cosine_similarity_loop, cosine_similarity, euclidean_distance_loop, euclidean_distance
from src.config import SEED, OUTPUT_DIR  


def benchmark():
    # Loading images  
    data = tfds.load("lfw:0.1.1", split="train", as_supervised=True)

    # Extract images into a numpy array
    images = []
    for label, image in tfds.as_numpy(data):
        images.append(image)
    images = np.array(images)

    # Generate N random image pairs for robust benchmarking
    N = 200
    np.random.seed(42)
    pair_indices = np.random.choice(len(images), size=(N, 2), replace=True)
    
    # --- Cosine benchmark  --- #
    loop_times_cos = []
    numpy_times_cos = []
    loop_results_cos = []
    numpy_results_cos = []
    
    # Loop through each pair and benchmark both implementations
    for idx1, idx2 in pair_indices:
        img1, img2 = images[idx1], images[idx2] # Get the two images for the current pair
        loop_time, loop_result = cosine_similarity_loop(img1, img2) # Benchmark the loop implementation and store time and result
        numpy_time, numpy_result = cosine_similarity(img1, img2) # Benchmark the numpy implementation and store time and result
        loop_times_cos.append(loop_time)
        numpy_times_cos.append(numpy_time)
        loop_results_cos.append(loop_result)
        numpy_results_cos.append(numpy_result)
        assert np.isclose(loop_result, numpy_result), "Cosine results do not match!" # Correctness check
    
    # Calculate average times
    avg_loop_cos = np.mean(loop_times_cos)
    avg_numpy_cos = np.mean(numpy_times_cos)
    
    print("\nCosine Similarity (N=200)")
    print(f"Loop Time (mean): {avg_loop_cos:.5f} sec")
    print(f"Numpy Time (mean): {avg_numpy_cos:.5f} sec")
    print(f"Speedup: {avg_loop_cos / avg_numpy_cos:.2f}x")
    print("Cosine correctness verified")


    # --- Euclidean benchmark --- #
    loop_times_euc = []
    numpy_times_euc = []
    loop_results_euc = []
    numpy_results_euc = []
    
    # Loop through each pair and benchmark both implementations
    for idx1, idx2 in pair_indices:
        img1, img2 = images[idx1], images[idx2] # Get the two images for the current pair
        loop_time, loop_result = euclidean_distance_loop(img1, img2) # Benchmark the loop implementation and store time and result
        numpy_time, numpy_result = euclidean_distance(img1, img2) # Benchmark the numpy implementation and store time and result
        loop_times_euc.append(loop_time)
        numpy_times_euc.append(numpy_time)
        loop_results_euc.append(loop_result)
        numpy_results_euc.append(numpy_result)
        assert np.isclose(loop_result, numpy_result), "Euclidean results do not match!" # Correctness check
    
    # Calculate average times
    avg_loop_euc = np.mean(loop_times_euc)
    avg_numpy_euc = np.mean(numpy_times_euc)

    print("\nEuclidean Distance (N=200)")
    print(f"Loop Time (mean): {avg_loop_euc:.5f} sec")
    print(f"Numpy Time (mean): {avg_numpy_euc:.5f} sec")
    print(f"Speedup: {avg_loop_euc / avg_numpy_euc:.2f}x")
    print("Euclidean correctness verified")

    # Saving results to artifacts
    np.save(os.path.join(OUTPUT_DIR, "cosine_loop_times.npy"), np.array(loop_times_cos))
    np.save(os.path.join(OUTPUT_DIR, "cosine_numpy_times.npy"), np.array(numpy_times_cos))
    np.save(os.path.join(OUTPUT_DIR, "euclidean_loop_times.npy"), np.array(loop_times_euc))
    np.save(os.path.join(OUTPUT_DIR, "euclidean_numpy_times.npy"), np.array(numpy_times_euc))
    


'''if __name__ == "__main__":
    benchmark()'''
