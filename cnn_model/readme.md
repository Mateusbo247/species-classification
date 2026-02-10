This repository contains two independent deep learning pipelines for image classification using:

- **ResNet-50 (TensorFlow / Keras)**
- **ResNet-152 (PyTorch)**

Both models are trained using transfer learning on a dataset organized into `train`, `val`, and `test` folders with **15 classes**.

------------------------------------------------------------------------
# Project Structure

The dataset must follow this structure:

    dataset/
    │
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    │
    ├── val/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    │
    └── test/
        ├── class_1/
        ├── class_2/
        └── ...

Each subfolder must contain the corresponding images for that class.

------------------------------------------------------------------------
# 1️⃣ ResNet-50 (TensorFlow / Keras)

File:

    resnet-50-csv.py

## What This Script Does

-   Loads pretrained **ResNet-50 (ImageNet weights)**
-   Freezes convolutional layers
-   Adds a custom classification head
-   Applies data augmentation (training set)
-   Trains with Early Stopping
-   Evaluates on test set
-   Generates prediction probability CSV files for:
    -   Train
    -   Validation
    -   Test

## Outputs

-   `resnet50_model.h5`
-   `output_csv/train_predictions.csv`
-   `output_csv/validation_predictions.csv`
-   `output_csv/test_predictions.csv`
-   Confusion matrix plot
-   Classification report (printed)

## How to Run

Install dependencies:

``` bash
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy
```

Run:

``` bash
python resnet-50-csv.py
```

------------------------------------------------------------------------

# 2️⃣ ResNet-152 (PyTorch)

File:

    resnet-152.py

## What This Script Does

-   Loads pretrained **ResNet-152**
-   Freezes backbone layers
-   Replaces final fully connected layer
-   Trains classifier head
-   Implements early stopping
-   Saves best model
-   Evaluates on test set
-   Generates prediction probability CSV files
-   Exports training history and classification report

## Outputs

-   `best_model_resnet152.pt`
-   `training_history.csv`
-   `classification_report_test.csv`
-   `predictions/val_predictions.csv`
-   `predictions/test_predictions.csv`
-   Confusion matrix plot

## How to Run

Install dependencies:

``` bash
pip install torch torchvision scikit-learn matplotlib pandas numpy
```

Run:

``` bash
python resnet-152.py
```

