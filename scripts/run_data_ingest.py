import numpy as np
import tensorflow_datasets as tfds
import json
import os
from src.data_ingest import ingest_lfw

def main():
    labels, dataset_split = ingest_lfw()
    print(f"Number of images: {len(labels)}\nTrain size: {len(dataset_split['train'])}\nVal size: {len(dataset_split['val'])}\nTest size:{len(dataset_split['test'])}")

if __name__=="__main__":
    main()
