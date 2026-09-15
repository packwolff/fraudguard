# =========================================================
# FRAUDGUARD RISK ENGINE
# =========================================================

from anomaly import predict_anomaly


# =========================================================
# RISK SCORE
# =========================================================

def calculate_risk_score(
    fraud_probability,
    anomaly_score
):

    # Both inputs are between 0 and 1
    #
    # Fraud probability = 70%
    # Anomaly score     = 30%

    risk_score = (

        (
            fraud_probability
            * 0.70
        )

        +

        (
            anomaly_score
            * 0.30
        )

    ) * 100

    risk_score = min(
        risk_score,
        100
    )

    return round(
        risk_score,
        2
    )


# =========================================================
# RISK LEVEL
# =========================================================

def classify_risk(
    risk_score
):

    if risk_score >= 70:

        return "HIGH"

    elif risk_score >= 30:

        return "MEDIUM"

    else:

        return "LOW"


# =========================================================
# SUSPICIOUS TRANSACTION
# =========================================================

def identify_suspicious_transaction(
    fraud_probability,
    anomaly_score,
    anomaly_detected,
    risk_level
):

    # A transaction is suspicious if
    # any strong risk signal exists.

    if (
        fraud_probability >= 0.50
        or anomaly_score >= 0.70
        or anomaly_detected
        or risk_level == "HIGH"
    ):

        return True

    return False


# =========================================================
# EXPLAINABLE REASONS
# =========================================================

def generate_reasons(
    transaction,
    fraud_probability,
    anomaly_score,
    anomaly_detected
):

    reasons = []

    amount = transaction[
        "amount"
    ]

    old_balance = transaction[
        "old_balance"
    ]

    new_balance = transaction[
        "new_balance"
    ]

    hour = transaction[
        "hour"
    ]


    # -----------------------------------------------------
    # Fraud probability
    # -----------------------------------------------------

    if fraud_probability >= 0.70:

        reasons.append(
            "🔴 Random Forest assigns a high "
            "probability of fraud."
        )

    elif fraud_probability >= 0.30:

        reasons.append(
            "🟠 Random Forest identifies "
            "moderate fraud probability."
        )


    # -----------------------------------------------------
    # Large transaction
    # -----------------------------------------------------

    if amount >= 50000:

        reasons.append(
            "🟠 Large transaction amount detected."
        )


    # -----------------------------------------------------
    # Large percentage of balance
    # -----------------------------------------------------

    if old_balance > 0:

        amount_ratio = (
            amount /
            old_balance
        )

        if amount_ratio >= 0.80:

            reasons.append(
                "🟠 Transaction uses a large "
                "portion of the origin account balance."
            )


    # -----------------------------------------------------
    # Account emptied
    # -----------------------------------------------------

    if (
        old_balance > 0
        and new_balance == 0
    ):

        reasons.append(
            "🟠 Origin account balance becomes "
            "zero after the transaction."
        )


    # -----------------------------------------------------
    # Impossible transaction
    # -----------------------------------------------------

    if (
        old_balance > 0
        and amount > old_balance
    ):

        reasons.append(
            "🔴 Transaction amount exceeds "
            "the available origin balance."
        )


    # -----------------------------------------------------
    # Unusual time
    # -----------------------------------------------------

    if hour <= 5:

        reasons.append(
            "🟠 Transaction occurs during "
            "an unusual early-morning period."
        )


    # -----------------------------------------------------
    # Anomaly model
    # -----------------------------------------------------

    if anomaly_score >= 0.70:

        reasons.append(
            "🔴 Isolation Forest reports "
            "strongly unusual transaction behavior."
        )

    elif anomaly_score >= 0.40:

        reasons.append(
            "🟠 Isolation Forest detects "
            "moderately unusual behavior."
        )


    # -----------------------------------------------------
    # Explicit anomaly flag
    # -----------------------------------------------------

    if anomaly_detected:

        reasons.append(
            "🔴 Transaction was classified as "
            "an anomaly by Isolation Forest."
        )


    # -----------------------------------------------------
    # No reasons
    # -----------------------------------------------------

    if not reasons:

        reasons.append(
            "🟢 No major risk indicators detected."
        )


    return reasons


# =========================================================
# RECOMMENDATION
# =========================================================

def get_recommendation(
    risk_level
):

    if risk_level == "HIGH":

        return (
            "MANUAL REVIEW REQUIRED"
        )

    elif risk_level == "MEDIUM":

        return (
            "ADDITIONAL VERIFICATION RECOMMENDED"
        )

    else:

        return (
            "TRANSACTION CAN PROCEED"
        )


# =========================================================
# COMPLETE ANALYSIS
# =========================================================

def analyze_transaction(
    transaction,
    fraud_probability
):

    # -----------------------------------------------------
    # Isolation Forest
    # -----------------------------------------------------

    anomaly_result = (
        predict_anomaly(
            transaction
        )
    )

    anomaly_score = (
        anomaly_result[
            "anomaly_score"
        ]
    )

    anomaly_detected = (
        anomaly_result[
            "anomaly"
        ]
    )


    # -----------------------------------------------------
    # Risk score
    # -----------------------------------------------------

    risk_score = (
        calculate_risk_score(
            fraud_probability,
            anomaly_score
        )
    )


    # -----------------------------------------------------
    # Risk level
    # -----------------------------------------------------

    risk_level = (
        classify_risk(
            risk_score
        )
    )


    # -----------------------------------------------------
    # Suspicious transaction
    # -----------------------------------------------------

    suspicious = (
        identify_suspicious_transaction(
            fraud_probability,
            anomaly_score,
            anomaly_detected,
            risk_level
        )
    )


    # -----------------------------------------------------
    # Explanations
    # -----------------------------------------------------

    reasons = (
        generate_reasons(
            transaction,
            fraud_probability,
            anomaly_score,
            anomaly_detected
        )
    )


    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    recommendation = (
        get_recommendation(
            risk_level
        )
    )


    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    return {

        "fraud_probability":
            round(
                fraud_probability,
                4
            ),

        "anomaly_score":
            round(
                anomaly_score,
                4
            ),

        "anomaly":
            anomaly_detected,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "suspicious":
            suspicious,

        "reasons":
            reasons,

        "recommendation":
            recommendation
    }