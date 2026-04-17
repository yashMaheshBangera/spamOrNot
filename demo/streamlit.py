import streamlit as st
import pickle
import numpy as np 
from user_preprocess import preprocess_email

# Load the trained model
with open('hyperparameter_tuning/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to make predictions
st.title("Spam or Not - Machine Learning Model")
email_input = st.text_area("Enter the email to classify if it's spam or not.")

if st.button("Classify"):
    # Preprocess the email input
    email_features = preprocess_email(email_input)
    
    # Make prediction
    prediction = model.predict([email_features])

    # Display results
    if prediction[0] == 1:
        st.success("The email is classified as **Spam**!", icon="✅")
    else:
        st.warning("The email is classified as **Not Spam**.", icon="❌")

    