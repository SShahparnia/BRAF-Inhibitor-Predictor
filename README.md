
# BRAF Inhibitor Predictor

## Description

This project leverages machine learning techniques using scikit-learn and other essential libraries to predict potential small molecule inhibitors targeting BRAF protein mutations. The model is trained and evaluated using robust data processing, enabling efficient and accurate predictions for early-stage drug discovery. Additionally, a Flask app provides a user-friendly interface for interacting with the predictor.

## Requirements

```bash
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

### 1. Training the Model
Run the script to train the model:
```bash
python scripts/train.py
```

### 2. Running Molecule Detection
Execute the detection script to analyze data:
```bash
python scripts/detect.py
```

### 3. Generating Metrics
Generate evaluation metrics with:
```bash
python scripts/generate_metrics.py
```

### 4. Running the Flask App
Start the Flask server to interact with the model through a web interface:

1. Run the application:
   ```bash
   python scripts/app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
   This will load the web interface where you can upload input data and view predictions.

### Example Workflow:
1. Prepare your input data (stored in the `data/` directory).
2. Use the training script to build the model.
3. Run the Flask app to interact with the predictor.

## Project Structure

- **data/** - Directory containing datasets and input files.
- **models/** - Folder for storing trained models.
- **scripts/** - Contains Python scripts for training, detection, and metrics generation.
  - `train.py` - Script for training the model.
  - `detect.py` - Molecule detection script.
  - `generate_metrics.py` - Evaluation metrics generator.
  - `app.py` - Flask app providing a web interface.
- **templates/** - HTML templates for the Flask app.
- **LICENSE** - License file for the project.
- **README.md** - Documentation and usage guide.
- **requirements.txt** - List of dependencies.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [Flask](https://flask.palletsprojects.com/)

---

**Author:** SShahparnia
