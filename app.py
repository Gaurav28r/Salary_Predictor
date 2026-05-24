import streamlit as st
import pickle
import numpy as np

# --- 1. Load the Saved Model and Encoders ---
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data["model"]
encoders = data["encoders"]

# --- 2. Build the Streamlit UI ---
st.set_page_config(page_title="Salary Predictor", page_icon="💰")
st.title("Data Science Salary Predictor 📊")
st.write("Enter the employee details below to estimate their annual salary.")

# --- 3. User Inputs ---
# We use the encoders to show the original text categories in the dropdowns
exp_level = st.selectbox("Experience Level", encoders['experience_level'].classes_)
emp_type = st.selectbox("Employment Type", encoders['employment_type'].classes_)
job_title = st.selectbox("Job Title", encoders['job_title'].classes_)
company_location = st.selectbox("Company Location", encoders['company_location'].classes_)

# --- 4. Prediction Logic ---
if st.button("Predict Salary"):
    # Convert user text inputs back to the numbers the model understands
    X_new = np.array([[
        encoders['experience_level'].transform([exp_level])[0],
        encoders['employment_type'].transform([emp_type])[0],
        encoders['job_title'].transform([job_title])[0],
        encoders['company_location'].transform([company_location])[0]
    ]])
    
    # Make prediction
    predicted_salary = model.predict(X_new)
    
    # Display Result
    st.subheader(f"Estimated Annual Salary: ${predicted_salary[0]:,.2f}")