import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from risk_engine import analyze_transaction


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FRAUD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# LOAD RANDOM FOREST
# =========================================================

try:

    with open(
        FRAUD_MODEL_PATH,
        "rb"
    ) as f:

        model = pickle.load(f)


    with open(
        PREPROCESSOR_PATH,
        "rb"
    ) as f:

        preprocessor = pickle.load(f)


except FileNotFoundError:

    st.error(
        "❌ Fraud model files were not found."
    )

    st.code(
        """
models/
├── fraud_model.pkl
└── preprocessor.pkl
        """
    )

    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.7;
        margin-bottom: 25px;
    }

    .explanation {
        padding: 12px;
        margin: 6px 0;
        border-radius: 8px;
        background-color: rgba(128,128,128,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ FraudGuard AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Transaction Risk Detection & Analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SYSTEM OVERVIEW
# =========================================================

st.header(
    "📊 System Overview"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

col1.metric(
    "Fraud Detection",
    "Random Forest"
)

col2.metric(
    "Anomaly Detection",
    "Isolation Forest"
)

col3.metric(
    "Risk Levels",
    "3"
)

col4.metric(
    "Explainability",
    "Enabled"
)

st.divider()


# =========================================================
# TRANSACTION INPUT
# =========================================================

st.header(
    "🔍 Analyze Transaction"
)

st.caption(
    "Enter transaction details. FraudGuard combines "
    "fraud probability, behavioral anomaly detection "
    "and transaction-level risk indicators."
)


# =========================================================
# FIRST ROW
# =========================================================

col1, col2, col3 = (
    st.columns(3)
)


with col1:

    step = st.number_input(
        "Transaction Step",
        min_value=1,
        value=300
    )


with col2:

    transaction_type = (
        st.selectbox(
            "Transaction Type",
            [
                "TRANSFER",
                "CASH_OUT",
                "PAYMENT",
                "DEBIT",
                "CASH_IN"
            ]
        )
    )


with col3:

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        value=85000.0,
        step=1000.0
    )


# =========================================================
# SECOND ROW
# =========================================================

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    oldbalanceOrg = (
        st.number_input(
            "Origin Balance Before (₹)",
            min_value=0.0,
            value=92000.0,
            step=1000.0
        )
    )


with col2:

    newbalanceOrig = (
        st.number_input(
            "Origin Balance After (₹)",
            min_value=0.0,
            value=7000.0,
            step=1000.0
        )
    )


with col3:

    oldbalanceDest = (
        st.number_input(
            "Destination Balance Before (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )
    )


with col4:

    newbalanceDest = (
        st.number_input(
            "Destination Balance After (₹)",
            min_value=0.0,
            value=135000.0,
            step=1000.0
        )
    )


st.write("")


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = st.button(
    "🔎 ANALYZE TRANSACTION",
    use_container_width=True,
    type="primary"
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    try:

        # =================================================
        # TRANSACTION DATAFRAME
        # =================================================

        transaction_df = pd.DataFrame({

            "step": [
                step
            ],

            "type": [
                transaction_type
            ],

            "amount": [
                amount
            ],

            "oldbalanceOrg": [
                oldbalanceOrg
            ],

            "newbalanceOrig": [
                newbalanceOrig
            ],

            "oldbalanceDest": [
                oldbalanceDest
            ],

            "newbalanceDest": [
                newbalanceDest
            ],

            "isFlaggedFraud": [
                0
            ],

            "hour": [
                step % 24
            ],

            "amount_log": [
                np.log1p(amount)
            ]

        })


        # =================================================
        # RANDOM FOREST
        # =================================================

        processed_data = (
            preprocessor.transform(
                transaction_df
            )
        )


        prediction = (
            model.predict(
                processed_data
            )[0]
        )


        probability = (
            model.predict_proba(
                processed_data
            )[0][1]
        )


        # =================================================
        # TRANSACTION FOR RISK ENGINE
        # =================================================

        transaction = {

            "amount":
                amount,

            "hour":
                step % 24,

            "old_balance":
                oldbalanceOrg,

            "new_balance":
                newbalanceOrig
        }


        # =================================================
        # COMPLETE RISK ANALYSIS
        # =================================================

        result = (
            analyze_transaction(
                transaction,
                probability
            )
        )


        fraud_probability = (
            result[
                "fraud_probability"
            ]
        )

        anomaly_score = (
            result[
                "anomaly_score"
            ]
        )

        anomaly_detected = (
            result[
                "anomaly"
            ]
        )

        risk_score = (
            result[
                "risk_score"
            ]
        )

        risk_level = (
            result[
                "risk_level"
            ]
        )

        suspicious = (
            result[
                "suspicious"
            ]
        )

        reasons = (
            result[
                "reasons"
            ]
        )

        recommendation = (
            result[
                "recommendation"
            ]
        )


        # =================================================
        # RESULT HEADER
        # =================================================

        st.divider()

        st.header(
            "🚨 AI Analysis Result"
        )


        # =================================================
        # MAIN METRICS
        # =================================================

        col1, col2, col3, col4 = (
            st.columns(4)
        )


        col1.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )


        col2.metric(
            "Anomaly Score",
            f"{anomaly_score * 100:.2f}%"
        )


        col3.metric(
            "Risk Score",
            f"{risk_score:.1f}/100"
        )


        col4.metric(
            "Risk Level",
            risk_level
        )


        # =================================================
        # SUSPICIOUS STATUS
        # =================================================

        if suspicious:

            st.error(
                "🚨 SUSPICIOUS TRANSACTION"
            )

        else:

            st.success(
                "🟢 TRANSACTION NOT FLAGGED AS SUSPICIOUS"
            )


        # =================================================
        # FRAUD PREDICTION
        # =================================================

        st.subheader(
            "🎯 Fraud Classification"
        )


        if prediction == 1:

            st.error(
                "🔴 FRAUDULENT TRANSACTION DETECTED"
            )

        else:

            st.success(
                "🟢 TRANSACTION CLASSIFIED AS LEGITIMATE"
            )


        # =================================================
        # RISK LEVEL
        # =================================================

        if risk_level == "HIGH":

            st.error(
                "🚨 HIGH RISK"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "⚠️ MEDIUM RISK"
            )

        else:

            st.success(
                "🟢 LOW RISK"
            )


        # =================================================
        # RISK PROGRESS
        # =================================================

        st.subheader(
            "📈 Overall Risk Assessment"
        )

        st.progress(
            int(risk_score)
        )

        st.caption(
            f"Risk Score: {risk_score:.1f}/100"
        )


        # =================================================
        # EXPLAINABILITY
        # =================================================

        st.subheader(
            "🧠 Why did FraudGuard make this decision?"
        )


        for reason in reasons:

            st.markdown(
                f"""
                <div class="explanation">
                    {reason}
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # ANOMALY RESULT
        # =================================================

        st.subheader(
            "🔬 Behavioral Anomaly Detection"
        )


        if anomaly_detected:

            st.warning(
                "Isolation Forest detected behavior "
                "that is unusual compared with the "
                "training transaction population."
            )

        else:

            st.success(
                "Isolation Forest did not detect "
                "strongly unusual behavior."
            )


        # =================================================
        # MODEL ASSESSMENT
        # =================================================

        st.subheader(
            "🤖 AI Model Assessment"
        )


        st.write(
            f"Random Forest fraud probability: "
            f"**{fraud_probability * 100:.2f}%**"
        )


        st.write(
            f"Isolation Forest anomaly score: "
            f"**{anomaly_score * 100:.2f}%**"
        )


        st.write(
            "The final risk score uses a 70% fraud-model "
            "weight and a 30% anomaly-model weight."
        )


        # =================================================
        # RECOMMENDATION
        # =================================================

        st.subheader(
            "💡 Recommended Action"
        )


        if risk_level == "HIGH":

            st.error(
                "🚨 MANUAL REVIEW REQUIRED\n\n"
                "The transaction contains significant "
                "risk indicators and should be reviewed "
                "before processing."
            )


        elif risk_level == "MEDIUM":

            st.warning(
                "⚠️ ADDITIONAL VERIFICATION RECOMMENDED\n\n"
                "Verify the transaction before processing."
            )


        else:

            st.success(
                "✅ TRANSACTION CAN PROCEED\n\n"
                "No significant combined fraud and "
                "anomaly indicators were detected."
            )


        # =================================================
        # TRANSACTION SUMMARY
        # =================================================

        st.subheader(
            "📋 Transaction Summary"
        )


        summary = pd.DataFrame({

            "Parameter": [

                "Transaction Type",

                "Transaction Amount",

                "Transaction Step",

                "Transaction Hour",

                "Origin Balance Before",

                "Origin Balance After",

                "Destination Balance Before",

                "Destination Balance After",

                "Fraud Probability",

                "Anomaly Score",

                "Risk Score",

                "Risk Level",

                "Suspicious"

            ],

            "Value": [

                transaction_type,

                f"₹{amount:,.2f}",

                step,

                step % 24,

                f"₹{oldbalanceOrg:,.2f}",

                f"₹{newbalanceOrig:,.2f}",

                f"₹{oldbalanceDest:,.2f}",

                f"₹{newbalanceDest:,.2f}",

                f"{fraud_probability * 100:.2f}%",

                f"{anomaly_score * 100:.2f}%",

                f"{risk_score:.1f}/100",

                risk_level,

                "YES" if suspicious else "NO"

            ]

        })


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # TECHNICAL DETAILS
        # =================================================

        with st.expander(
            "🔬 Technical Details"
        ):

            st.write(
                "1. Transaction data is processed using "
                "the same preprocessing pipeline used "
                "by the trained fraud model."
            )

            st.write(
                "2. A Random Forest classifier predicts "
                "whether the transaction is fraudulent."
            )

            st.write(
                "3. The classifier provides a fraud "
                "probability using predict_proba()."
            )

            st.write(
                "4. An Isolation Forest evaluates whether "
                "the transaction behavior is anomalous."
            )

            st.write(
                "5. Fraud probability contributes 70% "
                "of the final risk score."
            )

            st.write(
                "6. Anomaly score contributes 30% "
                "of the final risk score."
            )

            st.write(
                "7. The resulting score is classified "
                "as LOW, MEDIUM or HIGH risk."
            )

            st.write(
                "8. The system identifies suspicious "
                "transactions and provides reasons "
                "and a recommended action."
            )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            "❌ Transaction analysis failed."
        )

        st.write(
            "Error details:"
        )

        st.code(
            str(e)
        )