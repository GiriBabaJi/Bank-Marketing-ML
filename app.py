import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Classification App")
st.write("This app allows you to download a sample test dataset, upload it, select a model, and evaluate its performance.")

# ==========================================================
# 📥 DOWNLOAD SAMPLE TEST DATA
# ==========================================================

st.subheader("📥 Download Sample Test Dataset")

try:
    full_data = pd.read_csv("bank-additional-full.csv", sep=';')
    sample_test = full_data.sample(frac=0.2, random_state=42)

    csv_sample = sample_test.to_csv(index=False, sep=';').encode("utf-8")

    st.download_button(
        label="Download Sample Test CSV",
        data=csv_sample,
        file_name="sample_test_data.csv",
        mime="text/csv"
    )

except:
    st.info("Dataset file not found in repository.")

st.markdown("---")

# ==========================================================
# 🤖 MODEL SELECTION
# ==========================================================

model_choice = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# ==========================================================
# 📤 FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Test CSV File (bank-additional-full.csv format)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Load scaler and training columns
        scaler = joblib.load("model/scaler.pkl")
        train_columns = joblib.load("model/train_columns.pkl")

        # Load uploaded data
        data = pd.read_csv(uploaded_file, sep=';')
        original_data = data.copy()

        # ==============================
        # PREPROCESSING (same as training)
        # ==============================

        data.replace("unknown", np.nan, inplace=True)

        for col in data.select_dtypes(include='object').columns:
            data[col].fillna(data[col].mode()[0], inplace=True)

        data["y"] = data["y"].map({"no": 0, "yes": 1})

        data = pd.get_dummies(data, drop_first=True)

        X = data.drop("y", axis=1)
        y = data["y"]

        X = X.reindex(columns=train_columns, fill_value=0)

        X_scaled = scaler.transform(X)

        # ==============================
        # MODEL LOADING & PREDICTION
        # ==============================

        model = joblib.load(f"model/{model_choice}.pkl")

        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        # ==============================
        # DISPLAY METRICS
        # ==============================

        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(accuracy_score(y, predictions), 4))
        col1.metric("Precision", round(precision_score(y, predictions), 4))

        col2.metric("Recall", round(recall_score(y, predictions), 4))
        col2.metric("F1 Score", round(f1_score(y, predictions), 4))

        col3.metric("MCC", round(matthews_corrcoef(y, predictions), 4))
        col3.metric("AUC", round(roc_auc_score(y, probabilities), 4))

        # ==============================
        # CONFUSION MATRIX
        # ==============================

        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y, predictions)
        st.write(cm)

        # ==============================
        # CLASSIFICATION REPORT
        # ==============================

        st.subheader("📋 Classification Report")
        st.text(classification_report(y, predictions))

        # ==============================
        # DOWNLOAD PREDICTIONS
        # ==============================

        st.subheader("📥 Download Predictions")

        original_data["Predicted_Label"] = predictions
        original_data["Predicted_Probability"] = probabilities

        csv_results = original_data.to_csv(index=False, sep=';').encode("utf-8")

        st.download_button(
            label="Download Test Data with Predictions",
            data=csv_results,
            file_name="predicted_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Error processing file. Please upload correct dataset format.")
        st.write(e)
