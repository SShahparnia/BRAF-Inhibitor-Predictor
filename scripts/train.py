# coding: utf-8

"""
Training script for SVM model on chemical compounds dataset.
Authors: Shervan Shahparnia, Jason Tobin, Miles Thames
"""

# Import necessary libraries
import pandas as pd
import pickle
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score
import time
import matplotlib.pyplot as plt
import os

def main():
    """
    Main function to load data, preprocess, train SVM model, and save the model and preprocessing objects.
    """
    # Start timer
    start_time = time.time()

    # Load the CSV file
    print("Loading dataset...")
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data/chemical_compounds.csv")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded. Total records: {len(data)}\n")

    # Prepare the features and target variable
    print("Preparing features and target variable...")
    X = data.drop(columns=["CID", "Class"])
    y = data["Class"]

    # Identify and drop non-numeric columns
    print("Cleaning non-numeric columns...")
    non_numeric_columns = X.select_dtypes(include=["object"]).columns
    X_cleaned = X.drop(columns=non_numeric_columns)

    # Handle missing values by filling with the median
    print("Handling missing values...")
    median_values = X_cleaned.median()
    X_filled = X_cleaned.fillna(median_values)

    # Standardize the features
    print("Standardizing the features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Apply PCA for dimensionality reduction
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.95, random_state=42)  # Retain 95% variance
    X_pca = pca.fit_transform(X_scaled)

    # Perform Stratified k-Fold Cross-Validation
    print("Performing Stratified k-Fold Cross-Validation (5 folds)...")
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)

    # Use StratifiedKFold for balanced splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm_model, X_pca, y, cv=skf, scoring="accuracy")

    print(f"Stratified CV scores: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.2f}")
    print(f"Standard deviation: {cv_scores.std():.2f}\n")

    # Evaluate additional metrics using cross_validate
    cv_results = cross_validate(
        svm_model, X_pca, y, cv=skf, scoring=["accuracy", "precision", "recall", "f1"]
    )

    # Print detailed cross-validation results
    print(f"Accuracy: {cv_results['test_accuracy'].mean():.2f}")
    print(f"Precision: {cv_results['test_precision'].mean():.2f}")
    print(f"Recall: {cv_results['test_recall'].mean():.2f}")
    print(f"F1-Score: {cv_results['test_f1'].mean():.2f}\n")

    # Split the data into training and testing sets (80/20)
    print("Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}\n")

    # Train the SVM model on the training set
    print("Training the SVM model... This may take a while.")
    svm_model.fit(X_train, y_train)
    print("Model training complete!\n")

    # Save the trained model, scaler, PCA, feature names, and median values
    print("Saving the model, scaler, PCA, feature names, and median values...")
    with open("../models/svm_model.pkl", "wb") as model_file:
        pickle.dump(svm_model, model_file)

    with open("../models/scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open("../models/pca.pkl", "wb") as pca_file:
        pickle.dump(pca, pca_file)

    with open("../models/feature_names.pkl", "wb") as feature_file:
        pickle.dump(X_filled.columns.tolist(), feature_file)

    with open("../models/median_values.pkl", "wb") as median_file:
        pickle.dump(median_values, median_file)

    # Evaluate the model's performance on the test set
    print(f"\nEvaluating the model...")
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted"
    )  # Weighted for class imbalance
    report = classification_report(
        y_test, y_pred, target_names=["Non-Inhibitor", "Inhibitor"]
    )

    # Print results
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTraining Complete in {elapsed_time:.2f} seconds.")
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    print(f"Model Precision on Test Set: {precision:.2f}\n")
    print("Classification Report:\n", report)

#####################################################################################
if __name__ == "__main__":
    main()
else:
    print("train.py : Is intended to be executed and not imported.")
