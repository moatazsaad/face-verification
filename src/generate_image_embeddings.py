# Import MobileFaceNet
import torch
from PIL import Image
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from src.config import SCORE_FUNCTION

app = FaceAnalysis(
    name="buffalo_s", # lightweight model, can be changed later, but keeping this for speed and resource constraints
    allowed_modules=["detection", "recognition"] # allowed modules for detection and face recognition
)
app.prepare(
    ctx_id=-1,              # change to 0 if you have CUDA working
    det_size=(256, 256)     # resizes input image to 256x256 before detection, can be changed to 320x320
)


# Preprocessing function:
# Converts input ndarray into uint RGB formatting
# This ensures that the face detection and recognition models receive images i a consistent format
def arr_to_uint8_rgb(arr):
    """
    Converts input array to uint8 RGB image.
    Handles float images and already-uint8 images safely.
    """
    arr = np.asarray(arr)

    if arr.dtype == np.uint8:
        out = arr
    else:
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            out = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            out = np.zeros_like(arr, dtype=np.uint8)

    # Ensure 3 channels
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)

    return out


# Returns a 512-d embedding or None if no face is detected
# face.embedding is expected to be a 512-dimensional vector
# InsightFace expects BGR input, so we have to convert from RGB to BGR before passing to the model
def get_embedding(image):
    img_rgb = arr_to_uint8_rgb(image)
    img_bgr = img_rgb[:, :, ::-1]

    faces = app.get(img_bgr)

    if len(faces) == 0:
        return None

    # keep largest detected face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.embedding.astype(np.float32)

    # normalize explicitly for cosine similarity stability
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb


# Compputes one embedding per image for speedup, since embeddings are reused accross many pairs
# Saves embeddings to disc
def precompute_embeddings(images, save_path=None):
    """
    Computes one embedding per image.
    This is the main speed fix.
    """
    embeddings = []
    for i, img in enumerate(images):
        # Tracker for reassurance purposes
        if i % 100 == 0:
            print(f"Embedding image {i}/{len(images)}")
        embeddings.append(get_embedding(img))

    # Store as object array because some entries may be None
    embeddings = np.array(embeddings, dtype=object)

    if save_path is not None:
        np.save(save_path, embeddings, allow_pickle=True)


    return embeddings

# Compute cosine similarity score between two embeddings
def cosine_score(emb1, emb2):
    if emb1 is None or emb2 is None:
        return -1.0
    return float(np.dot(emb1, emb2))  # already L2-normalized

# Compute scores for all pairs using precomputed embeddings, with progress tracking
def compute_scores_from_embeddings(embeddings, pairs):
    scores = np.empty(len(pairs), dtype=np.float32)

    for i, (idx1, idx2) in enumerate(pairs):
        if i % 10000 == 0:
            print(f"Scoring pair {i}/{len(pairs)}")
        if SCORE_FUNCTION == "cosine_with_embeddings":
            scores[i] = cosine_score(embeddings[idx1], embeddings[idx2])
        elif SCORE_FUNCTION == "euclidean_with_embeddings":
            if embeddings[idx1] is None or embeddings[idx2] is None:
                scores[i] = -1.0
            else:
                scores[i] = np.linalg.norm(embeddings[idx1] - embeddings[idx2])

    return scores

