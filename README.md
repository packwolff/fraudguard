How to Run
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/Fraudgaurd.git
cd Fraudgaurd
2. Install Dependencies

Install the required Python packages using:

pip install -r requirements.txt
3. Run the Application

Start the FraudGuard application using Streamlit:

streamlit run app.py

That's it. app.py is the main entry point of the project.

The application automatically handles the required functionality from the other Python modules. You do not need to run anomaly.py or risk_engine.py separately.

app.py → Main Streamlit application
anomaly.py → Handles anomaly detection functionality
risk_engine.py → Handles risk scoring and analysis
models/ → Contains the pre-trained machine learning models used by the application

Once Streamlit starts, it will provide a local URL in the terminal. Open that URL in your browser to access FraudGuard.

Machine Learning Models

FraudGuard uses pre-trained machine learning models stored in the models/ directory. These models are loaded by the application during execution and are used to generate fraud predictions and risk assessments.

Usage Flow
User Input
    ↓
Streamlit Application (app.py)
    ↓
Fraud Detection
    ↓
Anomaly Detection (anomaly.py)
    ↓
Risk Analysis (risk_engine.py)
    ↓
Final Fraud / Risk Assessment
Project Purpose

FraudGuard aims to provide an accessible fraud detection system that goes beyond a simple fraud/not-fraud prediction. By combining machine learning predictions with anomaly detection and risk analysis, the system can provide a broader assessment of potentially suspicious transactions.
