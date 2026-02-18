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

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "theme" not in st.session_state:
    st.session_state.theme = "Light Mode"

# ---------------- VALID USERS ----------------
VALID_USERS = {
    "admin": "churnsense",
    "divyant": "founder123"
}

# ---------------- LOGIN SCREEN ----------------
if not st.session_state.logged_in:

    st.title("🧠 ChurnSense AI")
    st.markdown("### Predict. Prevent. Retain.")
    st.markdown("---")

    st.subheader("🔐 Login to Access Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in VALID_USERS and password == VALID_USERS[username]:
            st.session_state.logged_in = True
            st.success("Login successful! Reloading...")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- DASHBOARD ----------------
else:

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("ChurnSense AI")
    st.sidebar.markdown("Predict. Prevent. Retain.")
    st.sidebar.markdown("---")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ---------------- THEME TOGGLE ----------------
    st.session_state.theme = st.sidebar.radio(
        "🎨 Select Theme",
        ["Light Mode", "Dark Mode"],
        index=0 if st.session_state.theme == "Light Mode" else 1
    )

    # ---------------- APPLY THEME ----------------
    if st.session_state.theme == "Dark Mode":
        st.markdown("""
            <style>
            .stApp {
                background-color: #0E1117;
                color: white;
            }
            .stMetric {
                background-color: #1f2937;
                padding: 20px;
                border-radius: 12px;
            }
            section[data-testid="stSidebar"] {
                background-color: #111827;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: white;
                color: black;
            }
            .stMetric {
                background-color: #f3f4f6;
                padding: 20px;
                border-radius: 12px;
            }
            section[data-testid="stSidebar"] {
                background-color: #f9fafb;
            }
            </style>
        """, unsafe_allow_html=True)

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

    st.title("📊 Customer Risk Dashboard")

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

        # -------- Distribution Chart --------
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

        # -------- Suggestion Engine --------
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

        # -------- Download --------
        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="churnsense_predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("Upload a CSV file to begin risk analysis.")
