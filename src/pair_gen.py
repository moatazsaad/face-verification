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
    np.random.shuffle(combined)
    
    return combined[:, :2], combined[:, 2]

def save_splits(labels, dataset_split):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved_splits = {}
    for split_name, indices in dataset_split.items():
        pairs, labels_arr = generate_pairs(labels, indices)
        saved_splits[split_name] = (pairs, labels_arr)

        # Save files
        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_pairs.npy"), pairs)
        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_labels.npy"), labels_arr)
        print(f"{split_name} pairs saved successfully in {OUTPUT_DIR}")

    return saved_splits


'''if __name__ == "__main__":
    labels, dataset_split = ingest_lfw()
    save_splits(labels, dataset_split)'''
 
