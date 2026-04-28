# System Card: Face Verification System

## 1. System Overview

This project implements a face verification system that compares two face images and predicts whether they belong to the same person.

The final system uses an embedding-based pipeline:

1. Load two input images.
2. Preprocess each image into the expected format.
3. Extract face embeddings using InsightFace `buffalo_s`.
4. Compute cosine similarity between the two embeddings.
5. Compare the score against the operating threshold.
6. Return a match/non-match decision, confidence score, and latency values.

## 2. Intended Use

This system is intended for educational and experimental face verification use. It is designed to demonstrate a complete machine learning pipeline, including deterministic data handling, embedding-based inference, thresholding, confidence scoring, Docker packaging, profiling, and reproducibility.

The system may be used to test whether two face images are likely to show the same person under controlled project conditions.

## 3. Out-of-Scope Uses

This system should not be used for high-stakes or real-world identity decisions, including:

- law enforcement
- immigration
- banking or financial access
- employment screening
- school discipline
- surveillance
- access control without human review

The system was built as a course project and has not been validated for production or high-risk use.

## 4. Data Summary

The project uses the Labeled Faces in the Wild (LFW) dataset. The dataset contains face images collected under real-world conditions, including variation in lighting, pose, image quality, and facial expression.

The dataset is useful for educational face verification experiments, but it has limitations. It may not represent all demographic groups equally, and this project does not include reliable demographic metadata for full subgroup fairness evaluation.

## 5. Final Model and Threshold

- Embedding model: InsightFace `buffalo_s`
- Similarity score: cosine similarity on normalized embeddings
- Score type: `cosine_with_embeddings`
- Operating threshold: `0.29`

A pair is classified as a match when the cosine similarity score is greater than or equal to `0.29`.

## 6. Key Metrics

The following metrics are reported for the final embedding-based system using the same operating threshold used by the CLI.


- Threshold: `0.29`
- Accuracy: `0.99944`
- Precision: `1.00000`
- Recall / TPR: `0.99774`
- Specificity: `1.00000`
- F1 score: `0.99887`
- False positive rate / FPR: `0.00000`
- Balanced accuracy: `0.99887`
- True positives: `2209`
- True negatives: `6642`
- False positives: `0`
- False negatives: `5`

## 7. Confidence Score

The CLI reports a confidence value between 0 and 1. This confidence is not a probability. It is a margin-based score that shows how far the similarity score is from the operating threshold.

A higher confidence value means the score is farther from the threshold and the decision is clearer. A lower confidence value means the score is closer to the threshold and the decision is less certain.

## 8. Failure Modes and Limitations

The system may be unreliable when:

- no face is detected
- the image is blurry or low-resolution
- the face is partially covered
- lighting is poor
- the face is turned far away from the camera
- multiple faces appear in the image
- two different people look visually similar
- the input image is very different from the type of images seen in LFW

The system depends on successful face detection before embedding extraction. If face detection fails, the system may return a non-match or an unreliable result rather than a meaningful face comparison.

When multiple faces appear in an image, the system selects the largest detected face, but this may not always be the intended person.

## 9. Fairness and Misuse Risks

Because this project does not use reliable demographic metadata, it cannot claim equal performance across demographic groups.

Possible fairness risks include uneven performance due to differences in lighting, camera quality, pose, occlusion, age, skin tone, or other visual conditions. These risks are important because face verification systems can cause harm if used in high stakes settings.

The system should not be used as the only basis for identity decisions. Any real world use would require stronger validation, subgroup evaluation, monitoring and human review.

## 10. Operational Constraints

The final system was profiled on CPU for the required baseline.

Operational assumptions:

- Input images should contain a clear face.
- Images should be readable by PIL.
- The final CLI path should be run using the documented commands.
- The operating threshold must stay aligned with `src/config.py`.
- The system should be tested from a clean clone before final tagging.

## 11. Profiling Summary

CPU profiling showed that embedding generation is the main runtime cost. Preprocessing and scoring are very small compared with embedding extraction.

The profiling report is located at:

`reports/profiling_report.md`

The profiling CSV is located at:

`reports/profiling_cpu_summary.csv`

## 12. Reproducibility

The final release is reproducible from a clean clone using the README and reproducibility checklist.

Important files:

- README: `README.md`
- System Card: `reports/system_card.md`
- Profiling Report: `reports/profiling_report.md`
- Reproducibility Checklist: `reports/reproducibility_checklist.md`
- Final config: `src/config.py`

Final Git tag:

`v1.0-final`