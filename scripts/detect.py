# coding: utf-8

"""
Detection script for predicting inhibitor status using a trained SVM model.
Authors: Shervan Shahparnia, Jason Tobin, Miles Thames
"""

# Import necessary libraries
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os

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

def detect_inhibitor(input_data, svm_model, scaler, pca, feature_names, median_values):
    """
    Detect inhibitor status based on input data.
    Args:
        input_data (dict): Input data for prediction.
        svm_model: Trained SVM model.
        scaler: StandardScaler object.
        pca: PCA object.
        feature_names (list): List of feature names.
        median_values (pd.Series): Median values for missing data imputation.
    Returns:
        str: Prediction result ("Inhibitor" or "Non-Inhibitor").
    """
    # Ensure input is a DataFrame with the correct feature names
    df = pd.DataFrame([input_data], columns=feature_names)

    # Drop non-numeric columns if any
    non_numeric_columns = df.select_dtypes(include=["object"]).columns
    df_cleaned = df.drop(columns=non_numeric_columns)

    # Handle missing values by filling with the median
    df_filled = df_cleaned.fillna(median_values)

    # Standardize the features
    df_scaled = scaler.transform(df_filled)

    # Apply PCA transformation
    df_pca = pca.transform(df_scaled)

    # Make prediction
    prediction = svm_model.predict(df_pca)[0]
    result = "Inhibitor" if prediction == 1 else "Non-Inhibitor"

    return result

def main():
    """
    Main function to load models and perform inhibitor detection on example input.
    """
    # Load models and preprocessing files
    svm_model, scaler, pca, feature_names, median_values = load_models()

    # Example input data based on the CSV structure
    example_input = {
        "PUBCHEM_XLOGP3_AA": 5,
        "PUBCHEM_EXACT_MASS": 489.0725466,
        "PUBCHEM_MOLECULAR_WEIGHT": 489.9,
        "PUBCHEM_CACTVS_TPSA": 100,
        "PUBCHEM_MONOISOTOPIC_WEIGHT": 489.0725466,
        "PUBCHEM_TOTAL_CHARGE": 0,
        "PUBCHEM_HEAVY_ATOM_COUNT": 33,
        "PUBCHEM_ATOM_DEF_STEREO_COUNT": 0,
        "PUBCHEM_ATOM_UDEF_STEREO_COUNT": 0,
        "PUBCHEM_BOND_DEF_STEREO_COUNT": 0,
        "PUBCHEM_BOND_UDEF_STEREO_COUNT": 0,
        "PUBCHEM_ISOTOPIC_ATOM_COUNT": 0,
        "PUBCHEM_COMPONENT_COUNT": 1,
        "PUBCHEM_CACTVS_TAUTO_COUNT": 1,
        "PUBCHEM_COORDINATE_TYPE": 1,
    }

    print("Detecting inhibitor status...")
    result = detect_inhibitor(example_input, svm_model, scaler, pca, feature_names, median_values)
    print(f"Prediction: {result}")

#####################################################################################
if __name__ == "__main__":
    main()
else:
    print("detect.py : Is intended to be executed and not imported.")
