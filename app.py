import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
model    = joblib.load("model/gradient_boosting.pkl")
scaler   = joblib.load("model/scaler.pkl")
selector = joblib.load("model/selector.pkl")

feature_cols = [f"F{i+1}" for i in range(60)]
new_features = ["mean","std","max","min","range","energy","skewness","kurtosis"]
all_features = feature_cols + new_features

# ─────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────
def predict(data):
    df = pd.DataFrame(data).iloc[:, :60]
    df.columns = feature_cols

    # Feature Engineering
    df["mean"]     = df[feature_cols].mean(axis=1)
    df["std"]      = df[feature_cols].std(axis=1)
    df["max"]      = df[feature_cols].max(axis=1)
    df["min"]      = df[feature_cols].min(axis=1)
    df["range"]    = df["max"] - df["min"]
    df["energy"]   = (df[feature_cols] ** 2).sum(axis=1)
    df["skewness"] = df[feature_cols].skew(axis=1)
    df["kurtosis"] = df[feature_cols].kurt(axis=1)

    # Pipeline
    scaled   = scaler.transform(df[all_features].values)
    selected = selector.transform(scaled)
    pred     = model.predict(selected)
    prob     = model.predict_proba(selected)

    return pred, prob

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.set_page_config(page_title="Rock vs Mine", page_icon="🪨", layout="centered")

st.title("🪨 Rock vs Mine Classifier")
st.markdown("Upload a CSV file with **60 sonar features** to predict if it's a Rock or a Mine")
st.divider()

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    st.write(f"✅ File uploaded — {df.shape[0]} samples")
    st.dataframe(df.head())
    st.divider()

    if st.button("🔍 Predict", use_container_width=True):
        preds, probs = predict(df.values)

        for i, (pred, prob) in enumerate(zip(preds, probs)):
            label = "Mine 💣" if pred == 0 else "Rock 🪨"
            color = "🔴" if pred == 0 else "🟢"

            st.markdown(f"### Sample {i+1}: {color} {label}")
            col1, col2 = st.columns(2)
            col1.metric("Mine Probability", f"{prob[0]:.2%}")
            col2.metric("Rock Probability", f"{prob[1]:.2%}")
            st.progress(float(prob[1]))
            st.divider()