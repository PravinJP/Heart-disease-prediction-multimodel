import gradio as gr
import numpy as np
import joblib


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers



lr_model = joblib.load("models/lr_scaled_model.pkl")
dt_model = joblib.load("models/dt_scaled_model.pkl")
rf_model = joblib.load("models/rf_scaled_model.pkl")
svm_model = joblib.load("models/svm_scaled_model.pkl")
knn_model = joblib.load("models/knn_scaled_model.pkl")
xgb_model = joblib.load("models/xgb_scaled_model.pkl")

scaler = joblib.load("models/scaler.joblib")



def build_dl_model():
    model = Sequential()
    model.add(Input(shape=(11,)))

    model.add(Dense(
        16,
        activation="relu",
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l2(0.01)
    ))
    model.add(Dropout(0.5))

    model.add(Dense(
        8,
        activation="relu",
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l2(0.01)
    ))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.0004),
        loss="binary_crossentropy"
    )
    return model



dl_model = build_dl_model()
dl_model.load_weights(
    "models/dl_weights.weights.h5",
    skip_mismatch=True
)


sex_map = {"Female": 0, "Male": 1}

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
exang_map = {"No": 0, "Yes": 1}
slope_map = {"Up": 0, "Flat": 1, "Down": 2}



def predict_heart_disease(
    age, sex, cp, resting_bp, cholesterol, fasting_bs,
    rest_ecg, max_hr, exang, oldpeak, slope
):
    input_data = np.array([[ 
        age,
        sex_map[sex],
        cp_map[cp],
        resting_bp,
        cholesterol,
        fasting_bs,
        restecg_map[rest_ecg],
        max_hr,
        exang_map[exang],
        oldpeak,
        slope_map[slope]
    ]], dtype=float)

    input_scaled = scaler.transform(input_data)

    # ML predictions
    preds = [
        int(lr_model.predict(input_scaled)[0]),
        int(dt_model.predict(input_scaled)[0]),
        int(rf_model.predict(input_scaled)[0]),
        int(svm_model.predict(input_scaled)[0]),
        int(knn_model.predict(input_scaled)[0]),
        int(xgb_model.predict(input_scaled)[0]),
    ]

    # DL prediction
    dl_prob = float(dl_model.predict(input_scaled, verbose=0)[0][0])
    dl_pred = int(dl_prob >= 0.5)
    preds.append(dl_pred)

    # Majority voting
    final_pred = int(sum(preds) >= (len(preds) / 2))

    final_result = (
        " Heart Disease Detected"
        if final_pred == 1
        else "✅ No Heart Disease"
    )

    details = (
        f"Logistic Regression: {preds[0]}\n"
        f"Decision Tree: {preds[1]}\n"
        f"Random Forest: {preds[2]}\n"
        f"SVM: {preds[3]}\n"
        f"KNN: {preds[4]}\n"
        f"XGBoost: {preds[5]}\n"
        f"Deep Learning: {dl_pred} (prob={dl_prob:.2f})"
    )

    return final_result, details


interface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=45),
        gr.Dropdown(["Female", "Male"], label="Sex", value="Male"),
        gr.Dropdown(
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
            label="Chest Pain Type",
            value="Atypical Angina"
        ),
        gr.Number(label="Resting Blood Pressure", value=120),
        gr.Number(label="Cholesterol", value=200),
        gr.Dropdown([0, 1], label="Fasting Blood Sugar (1 = True)", value=0),
        gr.Dropdown(["Normal", "ST", "LVH"], label="Resting ECG", value="Normal"),
        gr.Number(label="Max Heart Rate Achieved", value=150),
        gr.Dropdown(["No", "Yes"], label="Exercise Induced Angina", value="No"),
        gr.Number(label="Oldpeak", value=1.0),
        gr.Dropdown(["Up", "Flat", "Down"], label="ST Slope", value="Flat"),
    ],
    outputs=[
        gr.Textbox(label="Final Prediction"),
        gr.Textbox(label="Model-wise Predictions", lines=8),
    ],
    title=" Heart Disease Prediction System",
    description="Multi-model heart disease prediction using ML + DL with ensemble voting.",
    allow_flagging="never",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="red",
        neutral_hue="slate"
    )
)




if __name__ == "__main__":
    interface.launch()
