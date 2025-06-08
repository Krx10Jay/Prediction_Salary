
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Page config
st.set_page_config(page_title="Salary Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model():
     return joblib.load( "model_142.pkl")

st.write("Loading model...")
reg = load_model()
st.write("Model loaded successfully.")

# Model metadata
MODEL_VERSION = "v1.0"
MODEL_DATE = "2024-06-01"

# Sidebar
st.sidebar.title(" Salary Predictor")
st.sidebar.markdown("Predict salaries based on years of experience.")
st.sidebar.markdown(f"**Model Version**: {MODEL_VERSION}")
st.sidebar.markdown(f"**Trained On**: {MODEL_DATE}")
# Main app title
st.title(" Predict Salary from Experience")
st.write("Choose one of the options below to estimate salaries:")

# Tabs: Single vs Batch prediction
tab1, tab2 = st.tabs([" Single Prediction", " Batch Prediction"])

# --- SINGLE PREDICTION ---
with tab1:
    st.subheader("Single Prediction")
    exp = st.slider("Years of Experience", 0, 42, 1)

    if st.button(" Predict Salary"):
        exp_array = np.array([[exp]])
        prediction = reg.predict(exp_array)[0]
        st.success(f" Estimated Salary: **${prediction:,.2f}**")

# --- BATCH PREDICTION ---
with tab2:
    st.subheader("Batch Prediction (Upload CSV)")
    st.markdown("**CSV Format:** Must have a column named `YearsExperience` with years of experience (integer/float).")

    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if "YearsExperience" not in df.columns:
                st.error("❌ CSV must contain a column named 'YearsExperience'.")
            else:
                exp_values = df["YearsExperience"].values.reshape(-1, 1)
                predictions = reg.predict(exp_values)
                df["Predicted_Salary"] = [f"${p:,.2f}" for p in predictions]

                st.success(" Batch predictions completed.")
                st.dataframe(df)

                # Downloadable CSV
                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button(" Download Results", csv_download, file_name="salary_predictions.csv", mime="text/csv")

        except Exception as e:

            st.error(f"Error: {e}")
