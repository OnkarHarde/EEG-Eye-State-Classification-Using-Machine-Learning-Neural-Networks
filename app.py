# =====================================================
# EEG EYE STATE CLASSIFICATION - END TO END APPLICATION
# Dataset: UCI EEG Eye State
# Models: Random Forest & Neural Network
# Deployment: Streamlit
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="EEG Eye State Classification",
    layout="wide"
)

DATA_PATH = "data/EEG_Eye_State.csv"
MODEL_DIR = "models"
RF_MODEL_PATH = f"{MODEL_DIR}/rf_model.pkl"
NN_MODEL_PATH = f"{MODEL_DIR}/nn_model.h5"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= DATA LOADING (FIXED) =================

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload EEG Eye State CSV",
    type=["csv"]
)

@st.cache_data
def load_data_from_path(path):
    return pd.read_csv(path)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found at `{DATA_PATH}`.\n\n"
            "Please upload the dataset using the sidebar."
        )
        st.stop()
    df = load_data_from_path(DATA_PATH)

# ------------------- DATA PREP -------------------
X = df.drop("eyeDetection", axis=1)
y = df["eyeDetection"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split_idx = int(len(X_scaled) * 0.8)

X_train = X_scaled[:split_idx]
X_test  = X_scaled[split_idx:]
y_train = y[:split_idx]
y_test  = y[split_idx:]

joblib.dump(scaler, SCALER_PATH)

# ------------------- RANDOM FOREST -------------------
if not os.path.exists(RF_MODEL_PATH):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_MODEL_PATH)
else:
    rf = joblib.load(RF_MODEL_PATH)

# ------------------- NEURAL NETWORK -------------------
if not os.path.exists(NN_MODEL_PATH):
    nn = Sequential([
        Dense(64, activation="relu", input_shape=(14,)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    nn.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    nn.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    nn.save(NN_MODEL_PATH)
else:
    nn = load_model(NN_MODEL_PATH)

# ------------------- EVALUATION -------------------
rf_pred = rf.predict(X_test)
nn_pred = (nn.predict(X_test) > 0.5).astype(int)

rf_acc = accuracy_score(y_test, rf_pred)
nn_acc = accuracy_score(y_test, nn_pred)

# ================= STREAMLIT UI =================

st.title("🧠 EEG Eye State Classification System")

st.markdown("""
**Dataset:** UCI EEG Eye State  
**Target:**  
- `1` → Eye Closed  
- `0` → Eye Open  
""")

# -------- SIDEBAR --------
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose Model",
    ["Random Forest", "Neural Network"]
)

st.sidebar.header("Accuracy")
st.sidebar.metric("Random Forest", f"{rf_acc:.2%}")
st.sidebar.metric("Neural Network", f"{nn_acc:.2%}")

# -------- INPUT --------
st.subheader("EEG Channel Inputs")

cols = st.columns(7)
values = []

for i in range(14):
    with cols[i % 7]:
        values.append(
            st.number_input(
                f"EEG {i+1}",
                value=0.0,
                format="%.5f"
            )
        )

X_input = scaler.transform(np.array(values).reshape(1, -1))

# -------- PREDICTION --------
if st.button("🔍 Predict Eye State"):
    if model_choice == "Random Forest":
        pred = rf.predict(X_input)[0]
        prob = rf.predict_proba(X_input)[0][pred]
    else:
        prob = nn.predict(X_input)[0][0]
        pred = int(prob > 0.5)

    label = "👁️ Eye Closed" if pred == 1 else "👀 Eye Open"

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{prob:.2f}**")

# -------- FEATURE IMPORTANCE --------
if model_choice == "Random Forest":
    st.subheader("Feature Importance (Random Forest)")
    fi_df = pd.DataFrame({
        "EEG Channel": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("EEG Channel"))

# -------- REPORT --------
with st.expander("📊 Classification Report"):
    if model_choice == "Random Forest":
        st.text(classification_report(y_test, rf_pred))
    else:
        st.text(classification_report(y_test, nn_pred))
