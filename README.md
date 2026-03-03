# ChurnSense AI 🚀  
### Predict. Prevent. Retain.

![Customer Risk Dashboard](dashboard-overview.png)

ChurnSense AI is a production-ready machine learning system that predicts customer churn risk and delivers actionable business insights through an interactive Streamlit dashboard.

---

## 🎯 Business Problem

Customer churn directly impacts revenue and long-term growth.  
Most organizations identify churn **after customers leave**, making retention reactive instead of proactive.

ChurnSense AI enables businesses to:

- Detect high-risk customers early  
- Segment customers by churn probability  
- Understand key churn-driving factors  
- Take data-driven retention actions  

---

## 🧠 End-to-End ML Pipeline

1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Model Training & Comparison  
5. Performance Evaluation  
6. Deployment via Interactive Dashboard  

---

## ⚙️ Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-Learn**
- **XGBoost**
- **Streamlit**
- **Matplotlib / Seaborn**

---

## 📊 Model Performance

Final Model: **XGBoost Classifier**

| Metric     | Score |
|------------|--------|
| Accuracy   | 87%    |
| Precision  | 0.84   |
| Recall     | 0.81   |
| ROC-AUC    | 0.89   |

The model balances precision and recall to effectively identify high-risk customers while minimizing false positives.

---

## 📸 Dashboard Highlights

### 🎯 Risk Segmentation
![Risk Segmentation](risk-segmentation.png)

Customers are categorized into:
- High Risk  
- Medium Risk  
- Low Risk  

This helps retention teams prioritize outreach efforts strategically.

---

### 🔍 Top Churn Drivers
![Top Churn Drivers](churn-drivers.png)

Feature importance analysis reveals major churn-driving factors such as:
- Tenure Months  
- Total Charges  
- Contract Type  
- Monthly Charges  

This enables data-driven business decisions instead of guesswork.

---

## 📈 Business Impact Simulation

If deployed in a telecom company with 10,000 customers:

- ~15% high-risk customers detected early  
- 8–10% potential improvement in retention  
- Significant annual revenue protection  
- Optimized retention budget allocation  

---

## 🚀 Live Demo

👉 Try the interactive application:  
https://churnsense-ai-e8uur4u5av4w7evxpz9eck.streamlit.app/

---

## 🔮 Future Enhancements

- SHAP-based model explainability  
- Hyperparameter tuning with Optuna  
- REST API deployment using FastAPI  
- Docker containerization  
- Real-time database integration  

---

## 📂 Project Structure
app/ # Streamlit dashboard
models/ # Trained ML models
data/ # Sample dataset
notebooks/ # EDA & model training workflow
requirements.txt

---

## 👨‍💻 Author

**Divyant Mayank**  
Machine Learning | Data Analytics | Aspiring Product Leader
