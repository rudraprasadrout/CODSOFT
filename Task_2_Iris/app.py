from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the HTML frontend

# ── Load the trained Naive Bayes model ──────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "NB_model.pkl")
model = joblib.load(MODEL_PATH)

# Species label map (matches notebook encoding)
LABEL_MAP = {
    1: "Iris-versicolor",
    2: "Iris-virginica",
    3: "Iris-setosa",
}

# Flower emoji / colour accents for fun API responses
FLOWER_META = {
    "Iris-setosa": {
        "emoji": "🌸",
        "description": "A small, sturdy iris found in Arctic regions. "
                       "Distinguished by its compact petals and vivid colours.",
        "color": "#e991b8",
    },
    "Iris-versicolor": {
        "emoji": "💜",
        "description": "The Blue Flag iris, native to North America. "
                       "Known for its striking violet-blue flowers.",
        "color": "#9b59b6",
    },
    "Iris-virginica": {
        "emoji": "🌺",
        "description": "Virginia iris, the largest of the three species. "
                       "Features deep-purple flowers and long, elegant petals.",
        "color": "#8e44ad",
    },
}


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "🌸 Iris Flower Classifier API is running",
        "model": "Gaussian Naive Bayes",
        "accuracy": "96.67%",
        "endpoints": {
            "POST /predict": "Classify an iris flower",
            "GET  /species": "List all species",
        },
    })


# ── List species ─────────────────────────────────────────────────────────────
@app.route("/species", methods=["GET"])
def species():
    return jsonify({
        "species": [
            {**{"name": name}, **meta}
            for name, meta in FLOWER_META.items()
        ]
    })


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate required keys
    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({
            "error": f"Missing fields: {missing}. "
                     f"Required: sepal_length, sepal_width, petal_length, petal_width"
        }), 400

    try:
        features = np.array([[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"]),
        ]])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid numeric values: {str(e)}"}), 400

    # Predict
    prediction_code = int(model.predict(features)[0])
    probabilities   = model.predict_proba(features)[0].tolist()
    species_name    = LABEL_MAP.get(prediction_code, "Unknown")
    meta            = FLOWER_META.get(species_name, {})

    # Build probability dict
    prob_dict = {
        LABEL_MAP[i + 1]: round(probabilities[i], 4)
        for i in range(len(probabilities))
    }

    confidence = round(max(probabilities) * 100, 2)

    return jsonify({
        "prediction":   species_name,
        "emoji":        meta.get("emoji", "🌸"),
        "description":  meta.get("description", ""),
        "color":        meta.get("color", "#1b0f20"),
        "confidence":   f"{confidence}%",
        "probabilities": prob_dict,
        "input": {
            "sepal_length": features[0][0],
            "sepal_width":  features[0][1],
            "petal_length": features[0][2],
            "petal_width":  features[0][3],
        },
    })


if __name__ == "__main__":
    print("🌸 Starting Iris Flower Classifier API …")
    print("   Running on  http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)