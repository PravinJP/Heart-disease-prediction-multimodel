

# ❤️ Heart Disease Prediction System (ML + DL Ensemble)

A **production-ready Heart Disease Prediction web application** that combines **multiple Machine Learning models and a Deep Learning neural network** using **ensemble voting**, deployed with an interactive **Gradio UI**.

This project demonstrates **end-to-end ML engineering skills** — from data preprocessing and model training to deployment-ready inference and UI integration.

---

## 🚀 Project Highlights

* 🔹 **Multi-Model Ensemble** (6 ML models + 1 DL model)
* 🔹 **Deep Learning with TensorFlow/Keras**
* 🔹 **Feature Scaling & Consistent Inference Pipeline**
* 🔹 **Model-wise Prediction Transparency**
* 🔹 **Clean, Interactive Web UI using Gradio**
* 🔹 **Production-safe model loading (no training-time dependencies)**

---

## 🧠 Models Used

### Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* XGBoost Classifier

### Deep Learning Model

* Fully Connected Neural Network (Dense NN)
* ReLU + Dropout + L2 Regularization
* Binary Classification (Sigmoid Output)

### Final Prediction

✔ **Ensemble Majority Voting** across all models for robust decision making

---

## 🧪 Input Features

The model predicts heart disease based on the following clinical parameters:

| Feature             | Description                                     |
| ------------------- | ----------------------------------------------- |
| Age                 | Patient age                                     |
| Sex                 | Male / Female                                   |
| Chest Pain Type     | Typical / Atypical / Non-Anginal / Asymptomatic |
| Resting BP          | Resting blood pressure                          |
| Cholesterol         | Serum cholesterol                               |
| Fasting Blood Sugar | >120 mg/dl (0 or 1)                             |
| Resting ECG         | Normal / ST / LVH                               |
| Max Heart Rate      | Maximum heart rate achieved                     |
| Exercise Angina     | Yes / No                                        |
| Oldpeak             | ST depression                                   |
| ST Slope            | Up / Flat / Down                                |

---

## 🖥️ Application UI

* Simple & clean medical-style interface
* Dropdowns for categorical inputs
* Numerical inputs with defaults
* Displays:

  * ✅ Final Heart Disease Prediction
  * 📊 Model-wise predictions for transparency

---

## ⚙️ Tech Stack

* **Python**
* **Scikit-learn**
* **TensorFlow / Keras**
* **XGBoost**
* **Gradio**
* **NumPy, Pandas, Joblib**

---

## 📂 Project Structure

```
Heart-disease-prediction/
│
├── app.py                  # Gradio web application
├── models/                 # Trained ML & DL models
│   ├── *.pkl
│   ├── scaler.joblib
│   └── dl_weights.weights.h5
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```

The app will launch in your browser 🎉

---

## 🎯 Why This Project Matters

✔ Shows **real-world ML deployment**, not just notebooks
✔ Demonstrates **model versioning & compatibility handling**
✔ Uses **ensemble learning for robustness**
✔ Focuses on **medical decision support** (high-impact domain)
✔ Clean architecture suitable for **production scaling**

---

## 🔮 Future Improvements

* Add SHAP / feature importance visualizations
* Deploy on Hugging Face / Cloud platform
* Add REST API endpoint
* Improve DL architecture with batch normalization

---

## 👤 Author

**Pravin J**
Aspiring Full-Stack & Machine Learning Engineer
📌 Passionate about building scalable, real-world AI systems

---

⭐ **If you’re a recruiter or engineer reviewing this project:**
This repository reflects **practical ML engineering**, **debugging resilience**, and **deployment readiness**, not just model accuracy.



