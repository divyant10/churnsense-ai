import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 ChurnSense AI")
st.sidebar.markdown("Predict. Prevent. Retain.")
st.sidebar.markdown("---")

# ---------------- LOAD MODEL ----------------
model_package = joblib.load("models/churn_model_package.pkl")
model = model_package["model"]
saved_threshold = model_package["threshold"]
feature_columns = model_package["feature_columns"]

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.sidebar.file_uploader("Upload Customer CSV", type=["csv"])

threshold = st.sidebar.slider(
    "Risk Threshold",
    0.1, 0.9,
    float(saved_threshold),
    0.05
)

# ---------------- MAIN TITLE ----------------
st.title("📊 Customer Risk Dashboard")
st.markdown("Upload your customer dataset to predict churn risk and take action.")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # -------- Preprocessing --------
    data_encoded = pd.get_dummies(data, drop_first=True)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)

    # -------- Prediction --------
    probs = model.predict_proba(data_encoded)[:, 1]
    data["Churn Probability"] = probs

    def segment(p):
        if p < threshold * 0.6:
            return "Low Risk"
        elif p < threshold:
            return "Medium Risk"
        else:
            return "High Risk"

    data["Risk Segment"] = data["Churn Probability"].apply(segment)

    # -------- KPI Section --------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(data))
    col2.metric("High Risk Customers",
                (data["Risk Segment"] == "High Risk").sum())
    col3.metric("Average Risk Score",
                round(data["Churn Probability"].mean(), 2))

    st.markdown("---")

    # -------- Probability Distribution --------
    st.subheader("📈 Probability Distribution")

    fig = px.histogram(
        data,
        x="Churn Probability",
        color="Risk Segment",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------- Risk Segmentation --------
    st.subheader("🎯 Risk Segmentation")

    segment_counts = data["Risk Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    pie_fig = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        hole=0.5
    )

    st.plotly_chart(pie_fig, use_container_width=True)

    # -------- AI Suggestions --------
    st.subheader("🧠 AI Retention Suggestions")

    high = (data["Risk Segment"] == "High Risk").sum()
    medium = (data["Risk Segment"] == "Medium Risk").sum()

    if high > 0:
        st.error("⚠ Immediate Action Required for High Risk Customers")
        st.write("- Assign retention manager")
        st.write("- Offer loyalty discount")
        st.write("- Call within 24 hours")

    if medium > 0:
        st.warning("📩 Engage Medium Risk Customers")
        st.write("- Send engagement email")
        st.write("- Offer plan review")
        st.write("- Provide feature tutorial")

    if high == 0 and medium == 0:
        st.success("✅ Customer base stable. No urgent intervention required.")

    # -------- Feature Importance --------
    if hasattr(model, "feature_importances_"):
        st.subheader("🔍 Top Churn Drivers")

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        bar_fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    # -------- High Risk Table --------
    st.subheader("🚨 High Risk Customers")
    high_risk = data[data["Risk Segment"] == "High Risk"]
    st.dataframe(high_risk.head(20))

    # -------- Download Button --------
    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="churnsense_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to begin risk analysis.")