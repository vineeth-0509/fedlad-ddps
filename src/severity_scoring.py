"""
severity_scoring.py
--------------------
This module defines logic for computing an Attack Severity Score (1–10)
based on the predicted attack type and model confidence probability.
"""

# ------------------------------------------------------------
# Function: calculate_severity()
# ------------------------------------------------------------
def calculate_severity(attack_type: str, confidence: float) -> float:
    """
    Calculate a severity score (1–10) for a detected attack.

    Parameters
    ----------
    attack_type : str
        The label predicted by the model (e.g., 'DDoS', 'PortScan', 'BENIGN', etc.)
    confidence : float
        The prediction probability or confidence score (0–1)

    Returns
    -------
    float
        Severity score between 1 and 10.
    """

    # Base severity mapping (higher = more critical)
    base_scores = {
        "DDoS": 9.5,
        "SYN": 8.5,
        "UDP": 7.5,
        "PortScan": 6.5,
        "Infiltration": 7.0,
        "WebAttack": 5.5,
        "BruteForce": 5.0,
        "BENIGN": 1.0
    }

    # Default to mid severity if label not recognized
    base_score = base_scores.get(attack_type, 5.0)

    # Confidence-weighted adjustment
    adjusted_score = base_score * (0.7 + 0.3 * confidence)

    # Clamp final score between 1 and 10
    severity = max(1.0, min(round(adjusted_score, 1), 10.0))

    return severity


# ------------------------------------------------------------
# Optional: Helper to format results neatly
# ------------------------------------------------------------
def format_prediction(label: str, confidence: float) -> dict:
    """
    Return a dictionary with label, confidence, and severity score.
    Useful for UI and API display.
    """
    severity = calculate_severity(label, confidence)
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "severity_score": severity
    }


# ------------------------------------------------------------
# Example usage (for testing)
# ------------------------------------------------------------
if __name__ == "__main__":
    test_preds = [
        ("DDoS", 0.95),
        ("UDP", 0.82),
        ("PortScan", 0.73),
        ("WebAttack", 0.60),
        ("BENIGN", 0.99)
    ]
    for attack, conf in test_preds:
        result = format_prediction(attack, conf)
        print(result)
