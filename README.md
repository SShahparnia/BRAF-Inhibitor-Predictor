# BRAF Inhibitor Predictor

## Description

This project leverages machine learning techniques using scikit-learn and other essential libraries to predict potential small molecule inhibitors targeting BRAF protein mutations. The model is trained and evaluated using robust data processing, enabling efficient and accurate predictions for early-stage drug discovery. Additionally, a Flask app provides a user-friendly interface for interacting with the predictor.

## Requirements

```
flask
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
- Flask-based web interface for predictions

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SShahparnia/BRAF-Inhibitor-Predictor.git
   cd BRAF-Inhibitor-Predictor
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
4. **Running the Flask App:**
   Start the Flask server to interact with the model through a web interface:
   ```bash
   python app.py
   ```
   Open your web browser and navigate to `http://127.0.0.1:5000` to use the application.

## Project Structure

- `train.py` - Model training pipeline
- `detect.py` - Molecule detection
- `generate_metrics.py` - Evaluation metrics generator
- `app.py` - Flask app providing a web interface
- `templates/` - HTML templates for the Flask app
- `static/` - Static files (e.g., CSS, JavaScript) for the Flask app

## License

MIT License

## Acknowledgments

- scikit-learn
- pandas
- matplotlib
- Flask

---

**Author:** SShahparnia
