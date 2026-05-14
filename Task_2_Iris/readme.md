# 🌸 Iris Flower Classification | CODSOFT Task 3

This repository contains the implementation for **Task 3** of the CODSOFT Machine Learning Internship. The project involves building a classification system that automatically identifies flower species based on their sepal and petal measurements.

---

### 🌐 Live Deployment

* **🚀 Demo Web App:** [Iris Classifier Interface](https://irisclassifiationrp.netlify.app/)
* **🔌 API Endpoint:** [Flask Backend on Render](https://www.google.com/search?q=https://iris-flower-classification-4oav.onrender.com/predict)

---

### 📖 Project Overview

The Iris dataset is one of the most famous datasets in pattern recognition. This project implements an end-to-end solution—from statistical analysis (EDA) to a functional web-based prediction tool.

**The Goal:** Predict the specific species of an Iris flower among three classes:

1. **Iris-Setosa** 🌸
2. **Iris-Versicolor** 💜
3. **Iris-Virginica** 🌺

### 🛠️ Key Implementation Details

* **Exploratory Data Analysis (EDA):** Performed comprehensive visualization using Seaborn and Matplotlib to identify feature correlations and species distributions.
* **Model Selection:** Evaluated multiple algorithms (Decision Trees, SVM, Random Forest). **Naive Bayes** was selected as the production model, achieving an impressive **96.67% accuracy**.
* **Production Pipeline:** The model was serialized using `joblib` and integrated into a Flask backend for real-time inference.
* **Dynamic UI:** A glassmorphism-inspired web interface that displays prediction confidence and species descriptions.

---

### ⚙️ Technical Stack

* **ML & Data:** Python, Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Backend:** Flask, Flask-CORS, Gunicorn
* **Frontend:** HTML5, CSS3 (Custom Design Tokens), JavaScript (Fetch API)

---

### 📂 Project Structure

```bash
task3_iris/
├── iris.ipynb          # Comprehensive EDA and model training
├── app.py              # Flask REST API for predictions
├── index.html          # Professional web interface
├── NB_model.pkl        # Trained Naive Bayes model
├── columns.pkl         # Feature column names for consistency
├── IRIS.csv            # Raw dataset
└── requirements.txt    # Production dependencies

```

---

### 🚀 Getting Started

1. **Clone the Repository:**
```bash
git clone https://github.com/rudraprasadrout/CODSOFT.git
cd Task_3_Iris_Classification

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Backend:**
```bash
python app.py

```


4. **Launch the UI:**
Open `index.html` in your browser to start classifying Iris flowers!

---

### 👨‍💻 Credits

**Developed by Rudra Prasad Rout**

* *Backend Architect & Aspiring Data Scientist*
* **GitHub:** [@rudraprasadrout](https://www.google.com/search?q=https://github.com/rudraprasadrout)
* **Email:** [routrp07@gmail.com]()

---

### 📝 Acknowledgments

Special thanks to **CODSOFT** for providing the platform to work on this machine learning task.

