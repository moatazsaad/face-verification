import tensorflow_datasets as tfds
import numpy as np
from src.similarity import cosine_similarity_loop, cosine_similarity, euclidean_distance_loop, euclidean_distance
from src.benchmark import benchmark

def main():
    benchmark()

if __name__ == "__main__":
    main()