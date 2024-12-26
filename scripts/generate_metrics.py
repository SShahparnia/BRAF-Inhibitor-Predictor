# coding: utf-8

"""
Script to generate training graphs, performance metrics, and confusion matrix for the SVM model.
Authors: Shervan Shahparnia, Jason Tobin, Miles Thames
"""

# Import necessary libraries
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

def load_models():
    """
    Load the saved models and preprocessing files.
    """
    print("Loading saved models and preprocessing files...")
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "../models/svm_model.pkl")
    scaler_path = os.path.join(script_dir, "../models/scaler.pkl")
    pca_path = os.path.join(script_dir, "../models/pca.pkl")
    feature_names_path = os.path.join(script_dir, "../models/feature_names.pkl")
    median_values_path = os.path.join(script_dir, "../models/median_values.pkl")

    with open(model_path, "rb") as model_file:
        svm_model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open(pca_path, "rb") as pca_file:
        pca = pickle.load(pca_file)
    with open(feature_names_path, "rb") as feature_file:
        feature_names = pickle.load(feature_file)
    with open(median_values_path, "rb") as median_file:
        median_values = pickle.load(median_file)

    print("Models and files loaded successfully.\n")
    return svm_model, scaler, pca, feature_names, median_values

def generate_metrics():
    """
    Generate training graphs, performance metrics, and confusion matrix for the SVM model.
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load models and preprocessing files
    svm_model, scaler, pca, feature_names, median_values = load_models()

    # Load the dataset
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data/chemical_compounds.csv")
    data = pd.read_csv(file_path)

    # Prepare the features and target variable
    X = data.drop(columns=["CID", "Class"])
    y = data["Class"]

    # Identify and drop non-numeric columns
    non_numeric_columns = X.select_dtypes(include=["object"]).columns
    X_cleaned = X.drop(columns=non_numeric_columns)

    # Handle missing values by filling with the median
    X_filled = X_cleaned.fillna(median_values)

    # Standardize the features
    X_scaled = scaler.transform(X_filled)

    # Apply PCA for dimensionality reduction
    X_pca = pca.transform(X_scaled)

    # Split the data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the SVM model on the training set
    svm_model.fit(X_train, y_train)

    # Evaluate the model's performance on the test set
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-Inhibitor", "Inhibitor"])

    # Print performance metrics
    print(f"Model Accuracy on Test Set: {accuracy * 100:.3f}%")
    print(f"Model Precision on Test Set: {precision * 100:.3f}%")
    print(f"Model Recall on Test Set: {recall * 100:.3f}%")
    print(f"Model F1-Score on Test Set: {f1 * 100:.3f}%\n")
    print("Classification Report:\n", report)

    # Save performance metrics to a text file
    with open(os.path.join(results_dir, "performance_metrics.txt"), "w") as file:
        file.write(f"Model Accuracy on Test Set: {accuracy * 100:.3f}%\n")
        file.write(f"Model Precision on Test Set: {precision * 100:.3f}%\n")
        file.write(f"Model Recall on Test Set: {recall * 100:.3f}%\n")
        file.write(f"Model F1-Score on Test Set: {f1 * 100:.3f}%\n\n")
        file.write("Classification Report:\n")
        file.write(report)

    # Save performance metrics summary as an image
    summary_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [f"{accuracy * 100:.3f}%", f"{precision * 100:.3f}%", f"{recall * 100:.3f}%", f"{f1 * 100:.3f}%"]
    }
    summary_df = pd.DataFrame(summary_data)
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    table = plt.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.title("Performance Metrics Summary", fontsize=14)
    plt.savefig(os.path.join(results_dir, "performance_summary.png"))
    plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=["Non-Inhibitor", "Inhibitor"], yticklabels=["Non-Inhibitor", "Inhibitor"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.show()

    # Plot training graphs (accuracy, precision, recall, f1-score)
    metrics = ["accuracy", "precision", "recall", "f1"]
    values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=metrics, y=values, palette="viridis")
    for i, v in enumerate(values):
        plt.text(i, v - 5, f"{v:.3f}%", ha='center', va='center', color='white', fontweight='bold')
    plt.ylim(0, 100)
    plt.title("Performance Metrics")
    plt.savefig(os.path.join(results_dir, "performance_metrics.png"))
    plt.show()

def main():
    """
    Main function to generate metrics and plots.
    """
    generate_metrics()

#####################################################################################
if __name__ == "__main__":
    main()
else:
    print("generate_metrics.py : Is intended to be executed and not imported.")