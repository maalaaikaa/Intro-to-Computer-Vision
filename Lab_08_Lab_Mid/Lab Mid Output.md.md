# Lab Mid Results — Registration No. 001

## Personal Settings

- Registration number: 001
- Random seed: 1
- HOG cell size: 16 × 16
- SIFT grid: 4 × 4
- SIFT keypoints: 16

## Dataset Setup

- Classes: eosinophil, monocyte, neutrophil, platelet
- Training images: 400 total (100 per class)
- Validation images: 120 total (30 per class)
- Test images: 120 total (30 per class)
- Processed image size: 128 × 128 grayscale

## HOG Feature Length

- Cells per side: 8
- Blocks per side: 7
- Total blocks: 49
- Features per block: 36
- **HOG feature length: 1,764**

## SIFT Feature Length

- Grid size: 4 × 4
- Keypoints: 16
- Descriptor length per keypoint: 128
- **SIFT feature length: 2,048**

## Robustness Test Settings

- Rotation: 20°
- Gaussian noise standard deviation: 0.05
- Models: HOG + SVM; SIFT + SVM
- SVM C values tested: 0.1, 1, 10

## Model Evaluation Results

Validation accuracies, macro-F1 scores, confusion matrices, test accuracies, robustness drops, and most-confused class pairs are **not available**: the uploaded notebook contains no saved cell outputs, and its BloodMNIST dataset was not available to execute the model here. These values are not fabricated.
