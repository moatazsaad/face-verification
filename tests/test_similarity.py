import os

import tensorflow_datasets as tfds
import numpy as np
from src.similarity import cosine_similarity_loop, cosine_similarity, euclidean_distance_loop, euclidean_distance
from src.config import SEED, OUTPUT_DIR  

# Test a simple similarity check between two identical images to verify correctness of both implementations
DATA_DIR = "data"   
os.makedirs(DATA_DIR, exist_ok=True)
data = tfds.load("lfw:0.1.1", split="train", as_supervised=True, data_dir=DATA_DIR)

# Extract images into a numpy array
images = []
for label, image in tfds.as_numpy(data):
    images.append(image)
images = np.array(images)

# Select two identical images (e.g., the first image twice)
img1 = images[0]
img2 = images[0]

# Test cosine similarity
loop_time_cos, loop_result_cos = cosine_similarity_loop(img1, img2)
numpy_time_cos, numpy_result_cos = cosine_similarity(img1, img2)
print("\nCosine Similarity Test")

print(f"Loop Result: {loop_result_cos:.5f}, Time: {loop_time_cos:.2f} sec")
print(f"Numpy Result: {numpy_result_cos:.5f}, Time: {numpy_time_cos:.5f} sec")
assert np.isclose(loop_result_cos, numpy_result_cos), "Cosine results do not match!"  # Correctness check
print("Cosine similarity correctness verified")

# Test euclidean distance
loop_time_euc, loop_result_euc = euclidean_distance_loop(img1, img2)
numpy_time_euc, numpy_result_euc = euclidean_distance(img1, img2)
print("\nEuclidean Distance Test")
print(f"Loop Result: {loop_result_euc:.5f}, Time: {loop_time_euc:.5f} sec")
print(f"Numpy Result: {numpy_result_euc:.5f}, Time: {numpy_time_euc:.5f} sec")
assert np.isclose(loop_result_euc, numpy_result_euc), "Euclidean results do not match!"  # Correctness check
print("Euclidean distance correctness verified")

# Run test using python3 -m tests.test_similarity