# Lab Mid Exam Results — HOG vs SIFT on Blood-Cell Images

**Course:** Computer Vision Lab  
**Registration No.:** 001  
**Student:** Malaika

## 1. Personal Settings

| Setting | Rule from registration number | Final value |
|---|---|---:|
| Random seed | Last two digits of `001` | 1 |
| HOG cell size | Last digit is odd | 16 × 16 pixels |
| SIFT grid | Last digit is between 0 and 4 | 4 × 4 grid |
| Number of fixed SIFT keypoints | 4 × 4 | 16 |

## 2. Dataset and Preprocessing

| Item | Final setting |
|---|---|
| Dataset | BloodMNIST, size 64 source images |
| Selected original class IDs | 1, 5, 6, 7 |
| Selected classes | Eosinophil, monocyte, neutrophil, platelet |
| New labels | 0, 1, 2, 3, respectively |
| Image preprocessing | RGB to grayscale, then resize |
| Final image size | 128 × 128 pixels |
| Random seed for sampling | 1 |

### Dataset split

| Split | Images per class | Number of classes | Total images | Status |
|---|---:|---:|---:|---|
| Training | 100 | 4 | 400 | Specified in notebook; execution not saved |
| Validation | 30 | 4 | 120 | Specified in notebook; execution not saved |
| Test | 30 | 4 | 120 | Specified in notebook; execution not saved |
| **Total** | **160** | **4** | **640** | Planned sample count |

### Class label mapping

| Original BloodMNIST label | Remapped label | Class name |
|---:|---:|---|
| 1 | 0 | Eosinophil |
| 5 | 1 | Monocyte |
| 6 | 2 | Neutrophil |
| 7 | 3 | Platelet |

## 3. HOG Feature Calculation

| Parameter | Calculation / value |
|---|---:|
| Image size | 128 × 128 |
| HOG cell size | 16 × 16 pixels |
| Cells per side | 128 ÷ 16 = 8 |
| Total cells | 8 × 8 = 64 |
| Cells per block | 2 × 2 |
| Blocks per side | 8 − 2 + 1 = 7 |
| Total blocks | 7 × 7 = 49 |
| Orientation bins per cell | 9 |
| Values per block | 2 × 2 × 9 = 36 |
| **HOG feature length per image** | **49 × 36 = 1,764** |
| Block normalisation | L2-Hys |

## 4. SIFT Feature Calculation

| Parameter | Final value |
|---|---:|
| Grid size | 4 × 4 |
| Number of keypoints | 16 |
| Keypoint size | 32 pixels |
| Grid centres on each axis | 16, 48, 80, 112 |
| Descriptor length per keypoint | 128 values |
| **SIFT feature length per image** | **16 × 128 = 2,048** |

### Feature comparison

| Property | HOG + SVM | Fixed-grid SIFT + SVM |
|---|---|---|
| Image input | 128 × 128 grayscale | 128 × 128 grayscale |
| Feature extraction | Histograms of oriented gradients | SIFT descriptors at fixed grid points |
| Feature length per image | 1,764 | 2,048 |
| Classifier | Linear SVM | Linear SVM |
| Feature scaling | StandardScaler | StandardScaler |
| Candidate C values | 0.1, 1, 10 | 0.1, 1, 10 |

## 5. SVM Hyperparameter Selection

The notebook selects the value of **C using validation accuracy only**. The test set is reserved for the final evaluation.

| Model | C values tested | Best C | Best validation accuracy | Validation macro-F1 |
|---|---|---|---|---|
| HOG + SVM | 0.1, 1, 10 | Not available — notebook has no saved execution output | Not available | Not available |
| Fixed-grid SIFT + SVM | 0.1, 1, 10 | Not available — notebook has no saved execution output | Not available | Not available |

**Best-performing model:** Cannot be determined from the uploaded notebook alone because its cells have not been executed or their outputs have not been saved. No model score has been estimated or invented.

## 6. Robustness Evaluation

The notebook evaluates the frozen models on the clean test images and on two altered versions of the same test images.

| Test condition | Applied change | Evaluation measure |
|---|---|---|
| Clean | No alteration | Test accuracy |
| Rotated | Rotate each image by 20° around its centre; black fill in corners | Test accuracy |
| Noisy | Add Gaussian noise with standard deviation 0.05 after scaling pixels to 0–1; clip to valid range | Test accuracy |

### Test accuracy comparison

| Model | Clean test accuracy | Rotated test accuracy | Noisy test accuracy |
|---|---:|---:|---:|
| HOG + SVM | Not available | Not available | Not available |
| Fixed-grid SIFT + SVM | Not available | Not available | Not available |

### Accuracy drop under image changes

Accuracy drop is calculated as **clean accuracy − altered-image accuracy**. A larger drop means the model was affected more by that alteration.

| Model | Rotation accuracy drop | Noise accuracy drop |
|---|---:|---:|
| HOG + SVM | Not available | Not available |
| Fixed-grid SIFT + SVM | Not available | Not available |
| Model with the larger drop | Cannot be determined without test results | Cannot be determined without test results |

### Predictions recorded before testing

| Condition | Prediction in the notebook | Reason given |
|---|---|---|
| Rotation | HOG + SVM would lose more accuracy | HOG uses edge directions in fixed cells; rotation changes edge directions and cell locations. |
| Noise | HOG + SVM would lose more accuracy | Noise adds small random gradients that may affect HOG histograms; SIFT uses Gaussian-weighted patches. |

**Were the predictions correct?** This cannot be verified until the actual clean, rotated, and noisy test accuracies are available.

## 7. Confusion-Matrix Results

The notebook intends to calculate validation confusion matrices for both models and identify the class pair most frequently confused in either direction.

| Model | Validation accuracy | Validation macro-F1 | Most-confused class pair | Mistakes in both directions combined |
|---|---:|---:|---|---:|
| HOG + SVM | Not available | Not available | Not available | Not available |
| Fixed-grid SIFT + SVM | Not available | Not available | Not available | Not available |

The actual confusion matrices and class-pair counts are not included because no saved cell outputs were present in the uploaded notebook.

## 8. Short Answers

### Why does HOG use block normalisation?
Block normalisation reduces the effect of lighting and contrast differences. It helps the descriptor represent the pattern of edge directions rather than simply the overall strength of the gradients.

### Why should C not be selected using the test set?
Choosing C on the test set would tune the model to the final evaluation data and could make the reported test accuracy over-optimistic. C should be selected using validation data, while the test set should be reserved for final evaluation.

### Why can ordinary SIFT keypoints be difficult to use directly with an SVM?
Ordinary SIFT can detect a different number of keypoints in each image. Since each keypoint has a 128-value descriptor, images can then have feature vectors of different lengths, which cannot be stacked directly into a fixed-width feature matrix. A fixed 4 × 4 grid provides 16 keypoints and a consistent 2,048-value feature vector per image.

### Why might grayscale images cause some blood-cell classes to be confused?
Grayscale removes colour information and keeps intensity and shape information. If two cell classes differ partly in the colour of their granules or cytoplasm, that difference is lost, potentially making the classes harder to distinguish.

## 9. AI Assistance Log

| # | Task / prompt summary | Change or decision | Verification described in the notebook |
|---:|---|---|---|
| 1 | Build the HOG + SVM and SIFT + SVM BloodMNIST lab for registration 001 | Seed 1, HOG cell 16 × 16, SIFT grid 4 × 4 | Compare settings with the registration-number rules |
| 2 | Select 100 / 30 / 30 images per class for classes 1, 5, 6, 7 | Remap class labels to 0–3 and use the fixed seed | Check split shapes and class counts |
| 3 | Calculate HOG feature length | HOG length calculated as 1,764 | Compare hand calculation with extracted feature length |
| 4 | Create a fixed 4 × 4 SIFT keypoint grid | 16 keypoints, size 32, centres 16, 48, 80, 112 | Check keypoint count and expected 2,048-value feature length |
| 5 | Apply rotation and noise to test images | Use the same altered images for both models | Visually compare clean, rotated, and noisy images |

## 10. Final Summary

| Result item | Final finding from the notebook specification |
|---|---|
| Registration number | 001 |
| Random seed | 1 |
| Training / validation / test sizes | 400 / 120 / 120 images planned |
| Number of selected classes | 4 |
| HOG feature length | 1,764 values per image |
| Fixed-grid SIFT feature length | 2,048 values per image |
| SVM C candidates | 0.1, 1, 10 |
| Rotation test | 20° |
| Noise test | Gaussian noise, σ = 0.05 |
| Best model by validation score | Not available without execution outputs |
| Best model on clean test data | Not available without execution outputs |
| Most robust model under rotation | Not available without execution outputs |
| Most robust model under noise | Not available without execution outputs |

**Data availability note:** The uploaded Jupyter notebook contains the experiment setup and result-producing steps, but its cells have no saved execution outputs. Therefore, the configuration, planned sample counts, hand-calculated feature lengths, and written explanations above are available from the notebook, while measured model scores, selected C values, confusion matrices, keypoint statistics, and robustness outcomes remain unavailable. These measured results have intentionally not been fabricated.
