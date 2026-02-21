import numpy as np                      
import json
import os                             
from collections import defaultdict     
from src.data_ingest import ingest_lfw
from src.config import SEED, OUTPUT_DIR                

def generate_pairs(labels, indices):
        
    np.random.seed(SEED)  

    label_with_indices = defaultdict(list) # Defaultdict to map each label to all indices where label occurs
    for index in indices:                          
        label = labels[index]
        label_with_indices[label].append(index)       # Append this index to the list of that label
        # for example label_with_indices ={"Juan":[0,2,4], Jong:[1,3,5]}

    # Generate positive pairs 
    pair_indices = []          # Will store tuples of 2 indices of the same label
    pair_label = []            # Will store 1 for same and 0 for different label

    for label, index_list in label_with_indices.items():                  # Iterate over each label and its list of indices
        if len(index_list) >= 2:                                          # Generate postive pairs for labels that has at least 2 indices in its value
            for i in range(len(index_list)):                              # Loop over indices
                for j in range(i + 1, len(index_list)):                   # Loop over this label starting from the i+1 index
                    pair_indices.append((index_list[i], index_list[j]))   # Append pair of indices(1st & 2nd then 1st and 3rd) of the same label
                    pair_label.append(1)                                  # Append 1 in positive pair list

    # Generate negative pairs 
    for i, label1 in enumerate(label_with_indices):                  # Loop over each label as the first label
        for label2 in list(label_with_indices.keys())[i + 1:]:       # Loop over labels after label1 to avoid duplicates
            for index1 in label_with_indices[label1]:                     # Loop over indices of the first label
                for index2 in label_with_indices[label2]:                 # Loop over indices of the second label
                    pair_indices.append((index1, index2))                 # Add the pair of indices to the list
                    pair_label.append(0)                                  # Append 0 in negative pair list



    # Convert to numpy  
    pair_indices = np.array(pair_indices)
    pair_label = np.array(pair_label)

    # Combine pairs and labels 
    combined = np.column_stack((pair_indices, pair_label))
    
    # Shuffle rows 
    np.random.shuffle(combined)

    # Split back into pairs and labels
    pair_indices, pair_label = combined[:, :2], combined[:, 2]
    
    return pair_indices, pair_label                                           

if __name__ == "__main__":

    labels, dataset_split = ingest_lfw()
    train_pairs, train_labels = generate_pairs(labels, dataset_split["train"])
    val_pairs, val_labels = generate_pairs(labels, dataset_split["val"])
    test_pairs, test_labels = generate_pairs(labels, dataset_split["test"])
 
    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "train_pairs.npy"), train_pairs)
    np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), train_labels)

    np.save(os.path.join(OUTPUT_DIR, "val_pairs.npy"), val_pairs)
    np.save(os.path.join(OUTPUT_DIR, "val_labels.npy"), val_labels)

    np.save(os.path.join(OUTPUT_DIR, "test_pairs.npy"), test_pairs)
    np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), test_labels)

    print("Splits saved successfully.")