import streamlit as st
import pickle
import numpy as np

# Load model
model_path = 'model/loan_default_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title("🏦 Loan Default Prediction App")
st.markdown("Fill in borrower info to predict loan default risk.")

# Inputs
loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
term = st.selectbox("Loan Term (months)", [360, 180, 240])

# Predict
if st.button("🔍 Predict Loan Default"):
    input_data = np.array([[loan_amount, interest_rate, income, credit_score, term]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of default (class 1)

    if prediction == 1:
        st.error("⚠️ Likely to **default** on the loan.")
    else:
        st.success("✅ Not likely to default.")

    st.info(f"📊 **Default Probability:** {probability*100:.2f}%")
