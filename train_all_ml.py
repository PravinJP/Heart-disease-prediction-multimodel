import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os

# =========================
# Load dataset
# =========================
df = pd.read_csv("heart.csv")

# =========================
# Encode categorical columns (MATCH APP)
# =========================
df["Sex"] = df["Sex"].map({"F": 0, "M": 1})

df["ChestPainType"] = df["ChestPainType"].map({
    "TA": 0,
    "ATA": 1,
    "NAP": 2,
    "ASY": 3
})

df["RestingECG"] = df["RestingECG"].map({
    "Normal": 0,
    "ST": 1,
    "LVH": 2
})

df["ExerciseAngina"] = df["ExerciseAngina"].map({
    "N": 0,
    "Y": 1
})

df["ST_Slope"] = df["ST_Slope"].map({
    "Up": 0,
    "Flat": 1,
    "Down": 2
})

# =========================
# Split features / target
# =========================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =========================
# Train models
# =========================
models = {
    "lr_scaled_model.pkl": LogisticRegression(max_iter=1000),
    "dt_scaled_model.pkl": DecisionTreeClassifier(),
    "rf_scaled_model.pkl": RandomForestClassifier(),
    "svm_scaled_model.pkl": SVC(probability=True),
    "knn_scaled_model.pkl": KNeighborsClassifier(),
    "xgb_scaled_model.pkl": XGBClassifier(
        eval_metric="logloss", use_label_encoder=False
    )
}

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"models/{name}")

joblib.dump(scaler, "models/scaler.joblib")

print("ALL ML MODELS SAVED CLEANLY")
