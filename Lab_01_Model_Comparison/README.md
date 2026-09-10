# Lab 01 — Transfer Learning Model Comparison for Skin Cancer Classification

## Overview

This lab focuses on multi-class skin lesion classification using transfer learning and deep learning. Pre-trained convolutional neural networks were trained and evaluated on a four-class subset of the ISIC Skin Cancer dataset.

The experiment compares different deep-learning models, classical machine-learning classifiers using deep features, and computational efficiency measures.

## Dataset

**Dataset:** Skin Cancer ISIC Dataset
**Source:** Kaggle — `nodoubttome/skin-cancer9-classesisic`

### Selected Classes

| Label | Class                |
| ----: | -------------------- |
|     0 | Melanoma             |
|     1 | Nevus                |
|     2 | Basal Cell Carcinoma |
|     3 | Actinic Keratosis    |

The training data is divided into:

* 80% training set
* 20% validation set
* Separate test set for final evaluation

## Models Compared

The following ImageNet-pretrained transfer-learning models are compared:

* AlexNet
* VGG16
* VGG19
* ResNet18
* ResNet50
* ResNet101
* DenseNet121
* EfficientNet-B0

## Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-Score
* Macro One-vs-Rest AUC Score

The project also compares computational efficiency using:

* Number of parameters
* Model size
* FLOPs
* Single-image inference time
* Test accuracy

## Deep Feature Classifier Comparison

Deep features are extracted using the trained EfficientNet-B0 model. These features are then used with the following classifiers:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Linear SVM
* RBF-SVM
* XGBoost

## Project Structure

```text
Lab_01_Model_Comparison/
│
├── CV_Lab_1.ipynb
├── README.md
├── output.md
│
├── results_4class/
│   ├── label_table.csv
│   ├── table1_transfer_models.csv
│   ├── table2_classifiers.csv
│   └── table3_efficiency.csv
│
└── results/
    └── graphs and visual outputs
```

## How to Run

1. Open `CV_Lab_1.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells from top to bottom.
4. The generated result tables are saved automatically in the `results_4class` folder.

## Output Files

| File                         | Description                                                         |
| ---------------------------- | ------------------------------------------------------------------- |
| `label_table.csv`            | Image count and label information for the selected classes          |
| `table1_transfer_models.csv` | Performance comparison of transfer-learning models                  |
| `table2_classifiers.csv`     | Performance comparison of classical classifiers using deep features |
| `table3_efficiency.csv`      | Computational efficiency comparison of deep-learning models         |
| `output.md`                  | Generated experimental output and results summary                   |

## Technologies Used

* Python
* PyTorch
* Torchvision
* Scikit-learn
* XGBoost
* KaggleHub
* Pandas and NumPy
* Google Colab

## Notes

* Image augmentation is applied to the training data to improve generalization.
* Class-weighted loss is used to reduce the impact of class imbalance.
* EfficientNet-B0 is used as the deep-feature extractor for classical classifier comparison.
* This project is for academic and experimental use only. It is not a medical diagnostic system.

