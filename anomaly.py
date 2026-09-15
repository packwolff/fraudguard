import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

ANOMALY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "anomaly_model.pkl"
)

ANOMALY_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "anomaly_scaler.pkl"
)


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "amount",
    "hour",
    "old_balance",
    "new_balance",
    "balance_change",
    "amount_ratio"
]


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_training_features(df):

    df = df.copy()

    # Convert PaySim step into simple hour
    df["hour"] = df["step"] % 24

    # Rename origin balance columns
    df["old_balance"] = df[
        "oldbalanceOrg"
    ]

    df["new_balance"] = df[
        "newbalanceOrig"
    ]

    # Balance movement
    df["balance_change"] = (
        df["old_balance"]
        -
        df["new_balance"]
    )

    # Transaction amount relative to balance
    df["amount_ratio"] = (
        df["amount"]
        /
        (df["old_balance"] + 1)
    )

    return df


# =========================================================
# TRAIN ISOLATION FOREST
# =========================================================

def train_anomaly_model(
    csv_path,
    rows=500000
):

    print("=" * 60)
    print("FRAUDGUARD - TRAINING ANOMALY MODEL")
    print("=" * 60)

    print("\nLoading dataset...")

    df = pd.read_csv(
        csv_path,
        nrows=rows
    )

    print(
        f"Loaded {len(df):,} transactions."
    )

    # Create features
    df = create_training_features(
        df
    )

    # Select features
    X = df[
        FEATURES
    ].copy()

    # Clean data
    X = X.replace(
        [np.inf, -np.inf],
        0
    )

    X = X.fillna(0)

    print("\nFeatures:")

    for feature in FEATURES:
        print(
            f"  ✓ {feature}"
        )

    print(
        "\nTraining Isolation Forest..."
    )

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    print(
        "✓ Isolation Forest training complete."
    )

    # Higher value = more anomalous
    raw_scores = -model.decision_function(
        X
    )

    # Normalize score
    scaler = MinMaxScaler()

    scaler.fit(
        raw_scores.reshape(-1, 1)
    )

    # Ensure models directory exists
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save model
    joblib.dump(
        model,
        ANOMALY_MODEL_PATH
    )

    # Save scaler
    joblib.dump(
        scaler,
        ANOMALY_SCALER_PATH
    )

    print(
        "\n✓ Saved:"
    )

    print(
        f"  {ANOMALY_MODEL_PATH}"
    )

    print(
        f"  {ANOMALY_SCALER_PATH}"
    )

    print("=" * 60)


# =========================================================
# PREDICT ANOMALY
# =========================================================

def predict_anomaly(
    transaction
):

    # Check files
    if not os.path.exists(
        ANOMALY_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Isolation Forest model not found: "
            f"{ANOMALY_MODEL_PATH}"
        )

    if not os.path.exists(
        ANOMALY_SCALER_PATH
    ):

        raise FileNotFoundError(
            "Anomaly scaler not found: "
            f"{ANOMALY_SCALER_PATH}"
        )

    # Load model
    model = joblib.load(
        ANOMALY_MODEL_PATH
    )

    scaler = joblib.load(
        ANOMALY_SCALER_PATH
    )

    # Convert transaction to DataFrame
    df = pd.DataFrame(
        [transaction]
    )

    # Derived features
    df["balance_change"] = (
        df["old_balance"]
        -
        df["new_balance"]
    )

    df["amount_ratio"] = (
        df["amount"]
        /
        (df["old_balance"] + 1)
    )

    # Select exact training features
    X = df[
        FEATURES
    ].copy()

    # Clean
    X = X.replace(
        [np.inf, -np.inf],
        0
    )

    X = X.fillna(0)

    # Isolation Forest prediction
    prediction = model.predict(
        X
    )[0]

    # Raw anomaly score
    raw_score = -model.decision_function(
        X
    )[0]

    # Normalize
    anomaly_score = scaler.transform(
        [[raw_score]]
    )[0][0]

    # Keep between 0 and 1
    anomaly_score = float(
        np.clip(
            anomaly_score,
            0,
            1
        )
    )

    return {

        "anomaly_score":
            anomaly_score,

        "anomaly":
            prediction == -1

    }


# =========================================================
# OPTIONAL COMMAND-LINE TRAINING
# =========================================================

if __name__ == "__main__":

    csv_path = os.path.join(
        BASE_DIR,
        "data",
        "paysim.csv"
    )

    train_anomaly_model(
        csv_path
    )

    test_transaction = {

        "amount": 85000,

        "hour": 3,

        "old_balance": 92000,

        "new_balance": 7000

    }

    result = predict_anomaly(
        test_transaction
    )

    print("\nTEST RESULT")
    print("-" * 40)

    print(
        "Anomaly Score:",
        result["anomaly_score"]
    )

    print(
        "Anomaly:",
        result["anomaly"]
    )