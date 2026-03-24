import numpy as np
import tensorflow_datasets as tfds
import json
import os
from src.data_ingest import ingest_lfw

labels, dataset_split = ingest_lfw()
if str(labels[0]) == "AJ_Cook" and str(labels[1]) == "AJ_Lamas":
    print("Test passed: First label is AJ_Cook")

# Run test using: python3 -m tests.test_data_ingest