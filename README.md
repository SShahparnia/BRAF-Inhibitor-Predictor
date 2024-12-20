# BRAF Inhibitor Predictor

## Description

This project leverages machine learning techniques using scikit-learn and other essential libraries to predict potential small molecule inhibitors targeting BRAF protein mutations. The model is trained and evaluated using robust data processing, enabling efficient and accurate predictions for early-stage drug discovery.

# Requirements

```
matplotlib
numpy
pandas
scikit-learn
seaborn
```

## Features

- Data preprocessing and scaling
- Machine learning model training and evaluation
- Performance metrics reporting

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SShahparnia/BRAF-Inhibitor-PredictorBRAF.git
   cd BRAF-Inhibitor-PredictorBRAF
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Training the Model:**
   ```bash
   python train.py
   ```
2. **Running Detection:**
   ```bash
   python detect.py
   ```
3. **Generating Metrics:**
   ```bash
   python generate_metrics.py
   ```

## Project Structure

- `train.py` - Model training pipeline
- `detect.py` - Molecule detection
- `generate_metrics.py` - Evaluation metrics generator

## License

MIT License

## Acknowledgments

- scikit-learn
- pandas
- matplotlib

---

**Author:** SShahparnia
