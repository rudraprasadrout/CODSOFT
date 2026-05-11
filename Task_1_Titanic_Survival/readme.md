# Titanic Survival Prediction API

Lightweight Flask API that serves a pre-trained scikit-learn model to predict Titanic passenger survival.

## Project structure

- `app.py` - Flask app exposing prediction endpoints
- `model.pkl` - pre-trained model (required to run the API)
- `Titanic-Dataset.csv` - dataset used for training (included)
- `titanic.ipynb` - training/EDA notebook
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.8 or newer
- `pip` available

## Install

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run

Start the API (development):

```bash
python app.py
```

By default the server listens on `http://0.0.0.0:5000`.

## API Endpoints

- `GET /` — root info
- `GET /health` — health check
- `POST /predict` — predict a single passenger. JSON body must include the fields: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `FamilySize`.
- `POST /predict/batch` — predict multiple passengers. Send a JSON array of passenger objects.

Example single prediction (curl):

```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass":3,"Sex":"male","Age":22,"Fare":7.25,"Embarked":"S","FamilySize":1}'
```

Example batch prediction:

```bash
curl -s -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"Pclass":3,"Sex":"male","Age":22,"Fare":7.25,"Embarked":"S","FamilySize":1}, {"Pclass":1,"Sex":"female","Age":38,"Fare":71.2833,"Embarked":"C","FamilySize":1}]'
```

## Model

The app expects a binary `model.pkl` file in the project root containing a fitted scikit-learn classifier with a `predict` and `predict_proba` API. If `model.pkl` is missing, you can train one using `titanic.ipynb` and export with `pickle.dump()`.

## Notes

- Input validation is performed by the API; invalid requests return HTTP 4xx with details.
- For production use, run behind a production WSGI server (e.g., `gunicorn`) and pin dependency versions in `requirements.txt`.

## License

MIT License — modify as needed.

## Contact

Open an issue or contact the maintainer for questions.
