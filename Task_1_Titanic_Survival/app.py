from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the model once at startup
# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

# Load the model using the absolute path
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: Could not find model.pkl at {model_path}")
    model = None
# ── helpers ──────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]

SEX_MAP     = {"male": 0, "female": 1}
EMBARKED_MAP = {"C": 0, "Q": 1, "S": 2}

def parse_input(data: dict):
    """Validate and convert raw JSON into the feature array the model expects."""
    errors = []

    # Check all fields present
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing field: '{field}'")
    if errors:
        return None, errors

    try:
        pclass = int(data["Pclass"])
        if pclass not in (1, 2, 3):
            errors.append("'Pclass' must be 1, 2, or 3")
    except (ValueError, TypeError):
        errors.append("'Pclass' must be an integer (1, 2 or 3)")
        pclass = None

    sex_raw = data["Sex"]
    if isinstance(sex_raw, str):
        sex = SEX_MAP.get(sex_raw.lower())
        if sex is None:
            errors.append("'Sex' must be 'male' or 'female'")
    elif sex_raw in (0, 1):
        sex = int(sex_raw)          # accept pre-encoded values too
    else:
        errors.append("'Sex' must be 'male' or 'female'")
        sex = None

    try:
        age = int(float(data["Age"]))
        if not (0 <= age <= 120):
            errors.append("'Age' must be between 0 and 120")
    except (ValueError, TypeError):
        errors.append("'Age' must be a number")
        age = None

    try:
        fare = float(data["Fare"])
        if fare < 0:
            errors.append("'Fare' must be non-negative")
    except (ValueError, TypeError):
        errors.append("'Fare' must be a number")
        fare = None

    embarked_raw = data["Embarked"]
    if isinstance(embarked_raw, str):
        embarked = EMBARKED_MAP.get(embarked_raw.upper())
        if embarked is None:
            errors.append("'Embarked' must be 'C', 'Q', or 'S'")
    elif embarked_raw in (0, 1, 2):
        embarked = int(embarked_raw)   # accept pre-encoded values too
    else:
        errors.append("'Embarked' must be 'C', 'Q', or 'S'")
        embarked = None

    try:
        family_size = int(data["FamilySize"])
        if family_size < 1:
            errors.append("'FamilySize' must be at least 1 (passenger counts as 1)")
    except (ValueError, TypeError):
        errors.append("'FamilySize' must be an integer")
        family_size = None

    if errors:
        return None, errors

    features = pd.DataFrame([[pclass, sex, age, fare, embarked, family_size]],
                             columns=["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"])
    return features, []


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Titanic Survival Prediction API",
        "endpoints": {
            "POST /predict": "Predict survival for a single passenger",
            "POST /predict/batch": "Predict survival for multiple passengers",
            "GET  /health": "Health check"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    features, errors = parse_input(data)
    if errors:
        return jsonify({"errors": errors}), 422

    prediction   = int(model.predict(features)[0])
    probability  = float(model.predict_proba(features)[0][1])

    return jsonify({
        "survived": bool(prediction),
        "survived_label": "Yes" if prediction == 1 else "No",
        "survival_probability": round(probability, 4),
        "input": data
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array of passengers"}), 400

    results = []
    for i, passenger in enumerate(data):
        features, errors = parse_input(passenger)
        if errors:
            results.append({"index": i, "errors": errors})
        else:
            prediction  = int(model.predict(features)[0])
            probability = float(model.predict_proba(features)[0][1])
            results.append({
                "index": i,
                "survived": bool(prediction),
                "survived_label": "Yes" if prediction == 1 else "No",
                "survival_probability": round(probability, 4)
            })

    return jsonify({"count": len(results), "results": results})


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)