import tensorflow_datasets as tfds
import numpy as np
from src.similarity import (cosine_similarity_loop, cosine_similarity, euclidean_distance_loop, euclidean_distance)


def benchmark():
    # Loading images  
    data = tfds.load("lfw:0.1.1", split="train", as_supervised=True)

    images = []
    for label, image in tfds.as_numpy(data):
        images.append(image)

    images = np.array(images)

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
    
if __name__ == "__main__":
    benchmark()
