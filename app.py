"""
Telecom Customer Churn Prediction - Streamlit App
--------------------------------------------------
This is a simple web app made using Streamlit.
The user fills in customer details, and the app predicts
whether that customer is likely to churn (leave) or not.

The model, scaler and encoders used here are the SAME ones
that were trained and saved in the Jupyter notebook, so the
predictions will match what we saw during model training.
"""

import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------
# Load the saved model, scaler, encoders and column order
# ---------------------------------------------------------
model = pickle.load(open("models/churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))

# ---------------------------------------------------------
# Basic page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📞")

st.title("📞 Telecom Customer Churn Prediction")
st.write(
    "Fill in the customer's details below and click **Predict** "
    "to check if the customer is likely to churn (leave the company)."
)

st.markdown("---")

# ---------------------------------------------------------
# Input widgets
# We group them into a few sections just to keep the page
# easy to read.
# ---------------------------------------------------------

st.subheader("1. Customer Info")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has a Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with col2:
    tenure = st.slider("Tenure (months with company)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

st.subheader("2. Phone & Internet Services")
col3, col4 = st.columns(2)

with col3:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col4:
    online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])

col5, col6 = st.columns(2)
with col5:
    tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
with col6:
    streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])

st.subheader("3. Account & Billing")
col7, col8 = st.columns(2)

with col7:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

with col8:
    payment_method = st.selectbox(
        "Payment Method",
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
    )

st.markdown("---")

# ---------------------------------------------------------
# When the user clicks Predict
# ---------------------------------------------------------
if st.button("🔍 Predict Churn"):

    # Step 1: put all the inputs into a dictionary
    # (the keys MUST match the column names used during training)
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Step 2: turn it into a single-row DataFrame
    input_df = pd.DataFrame([input_dict])

    # Step 3: encode the text columns using the SAME encoders
    # that were used when we trained the model
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Step 4: make sure the columns are in the exact same order
    # as during training
    input_df = input_df[columns]

    # Step 5: scale the input the same way the training data was scaled
    input_scaled = scaler.transform(input_df)

    # Step 6: predict!
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of churn (class 1)

    st.markdown("### Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn**.\n\nChurn probability: **{probability*100:.1f}%**")
    else:
        st.success(f"✅ This customer is **likely to stay**.\n\nChurn probability: **{probability*100:.1f}%**")

    # a simple progress bar to visualize the probability
    st.progress(float(probability))

st.markdown("---")
st.caption(
    "This app uses a Logistic Regression model trained on the IBM Telco Customer Churn dataset. "
    "Built as part of a college ML capstone project."
)
