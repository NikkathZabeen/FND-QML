# Fake News Detection using Classical and Quantum Machine Learning

This project implements a fake news detection system, specifically focusing on COVID-19 related content. It leverages both classical machine learning (SVM) and provides a foundation for Quantum Machine Learning (QML) approaches using PennyLane.

## Overview

The goal of this project is to distinguish between real and fake news articles/tweets. The current implementation uses the "Constraint" dataset, processes the text data, and trains Support Vector Machine (SVM) models. It also sets up the environment for Quantum Machine Learning experiments.

## Dataset

The project uses the **Constraint@AAAI2021 COVID-19 Fake News Detection Dataset**.
The notebook expects the following files to be located in your Google Drive at `/content/drive/MyDrive/CovidDataset/`:

-   `Constraint_English_Train.xlsx`
-   `Constraint_English_Val.xlsx`
-   `english_test_with_labels.xlsx`

## Dependencies

The project relies on the following Python packages:

-   **Quantum Computing:** `pennylane`, `pennylane-lightning`
-   **Machine Learning:** `scikit-learn`
-   **Data Processing:** `pandas`, `numpy`, `openpyxl`
-   **Visualization:** `seaborn`, `matplotlib`

## Project Workflow

The `fnd_qml.ipynb` notebook performs the following steps:

1.  **Environment Setup:** Mounts Google Drive to access the dataset and installs necessary libraries.
2.  **Data Loading & Preprocessing:**
    -   Loads training, validation, and test datasets.
    -   Cleans text (lowercase, removes URLs, mentions, special characters).
    -   Encodes labels (Real/Fake) to numerical values.
3.  **Feature Extraction:**
    -   Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.
    -   Configuration: N-grams (1,2), Max features: 5000.
4.  **Model Training & Evaluation:**
    -   **Linear SVC:** Trains a Linear Support Vector Classifier and evaluates it on validation and test sets.
    -   **RBF SVM:** Trains a Support Vector Machine with a Radial Basis Function (RBF) kernel on the full feature set.
5.  **Quantum Machine Learning (In Progress):**
    -   The environment is set up with PennyLane (`import pennylane as qml`) to implement Quantum Kernels or Quantum Neural Networks for classification.

## Usage

1.  **Upload Dataset:** Ensure the dataset files are uploaded to your Google Drive in the folder `CovidDataset`.
2.  **Open in Colab:** Upload `fnd_qml.ipynb` to Google Colab.
3.  **Run Cells:** Execute the cells sequentially.
    -   You will be asked to authenticate and mount your Google Drive.
    -   The notebook will install dependencies, process the data, and train the models.

## Results

The notebook outputs classification reports (Precision, Recall, F1-Score) and confusion matrices for both validation and test sets to evaluate the performance of the models.
