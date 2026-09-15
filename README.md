# FraudGuard

FraudGuard is a machine learning-based fraud detection system designed to identify potentially fraudulent transactions and assess their risk.

The project combines **fraud detection, anomaly detection, and risk analysis** to provide a more comprehensive evaluation of transactions. The trained machine learning models are stored in the `models/` folder and are automatically loaded by the application.

## Features

- Machine learning-based fraud detection
- Anomaly detection for identifying unusual transactions
- Risk scoring and analysis
- Uses pre-trained machine learning models
- Interactive Streamlit interface
- Automated integration of fraud, anomaly, and risk analysis

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Machine Learning

## How to Run

### 1. Clone the Repository

    git clone https://github.com/YOUR_USERNAME/Fraudgaurd.git
    cd Fraudgaurd

### 2. Install Dependencies

    pip install -r requirements.txt

### 3. Run the Application

    streamlit run app.py

That's it! `app.py` is the main entry point of the project.

The application automatically imports and uses the functionality from:

- `anomaly.py` — Anomaly detection
- `risk_engine.py` — Risk scoring and analysis
- `models/` — Pre-trained machine learning models

You do **not** need to run `anomaly.py` or `risk_engine.py` separately.

Once Streamlit starts, open the local URL provided in the terminal to access FraudGuard.
