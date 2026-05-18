import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  

MODEL_PATH = 'GB_model.pkl'
COLUMNS_PATH = 'columns.pkl'

# Load the model and expected columns during startup
try:
    model = joblib.load(MODEL_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"Warning during startup load: {e}")
    model = None
    expected_columns = ['TV', 'Radio', 'Newspaper']

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the Sales Prediction API!",
        "status": "running" if model is not None else "model_load_failed",
        "endpoints": {
            "/predict": "POST - Submit feature values to get sales predictions"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    # If model failed to load due to library versions, we try to reload or handle it
    global model
    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            return jsonify({
                "error": f"Model is not available or loaded on the server due to an environment mismatch: {str(e)}. "
                         "Ensure scikit-learn versions match between training and deployment environments."
            }), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request. Please provide JSON input."}), 400

    is_batch = isinstance(data, list)
    records = data if is_batch else [data]

    predictions = []
    
    for idx, record in enumerate(records):
        missing_fields = [col for col in expected_columns if col not in record]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "record_index": idx if is_batch else None
            }), 400
        
        try:
            features = {col: float(record[col]) for col in expected_columns}
        except (ValueError, TypeError):
            return jsonify({
                "error": "All input values (TV, Radio, Newspaper) must be numeric.",
                "record_index": idx if is_batch else None
            }), 400
            
        input_df = pd.DataFrame([features], columns=expected_columns)
        
        try:
            pred = model.predict(input_df)[0]
            predictions.append(float(pred))
        except Exception as e:
            return jsonify({
                "error": f"Prediction failed: {str(e)}",
                "record_index": idx if is_batch else None
            }), 500

    if is_batch:
        return jsonify({"predictions": predictions})
    else:
        return jsonify({"prediction": predictions[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)