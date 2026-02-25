import numpy as np
import tensorflow_datasets as tfds
import json
import os
from src.data_ingest import ingest_lfw
from src.pair_gen import save_splits, generate_pairs


labels = ['A', 'A', 'B', 'B', 'C']
indices = [0, 1, 2, 3, 4]
pairs, pair_labels = generate_pairs(labels, indices) # Generate the pairs and pair labels
print(pair_labels)

bool = True
for i in range(2):
    new_pairs, new_pair_labels = generate_pairs(labels, indices) # Generate the pairs and pair labels again to check if they are same as before
    if np.array_equal(pairs, new_pairs) and np.array_equal(pair_labels, new_pair_labels):
        continue
    else:
        bool = False
        print("Test failed: Pairs and labels are not consistent across runs.")
        break

if bool:
    print("Test passed: Pairs and labels are consistent across runs.")
else:
    print("Test failed: Pairs and labels are not consistent across runs.")
        

# python3 -m tests.test_pair_generation