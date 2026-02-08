import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("goal_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Football Goals Prediction App")
st.write("Predict total number of goals based on shooting statistics in 2012 tournament")

# Input fields
shots_on_target = st.number_input("Shots on target", min_value=0.0)
shots_off_target = st.number_input("Shots off target", min_value=0.0)
shooting_accuracy = st.number_input("Shooting Accuracy (%)", min_value=0.0)
goals_to_shots = st.number_input("Goals to Shots (%)", min_value=0.0)
total_shots = st.number_input("Total shots (including blocked)", min_value=0.0)

if st.button("Predict Goals"):
    input_data = np.array([[shots_on_target,
                             shots_off_target,
                             shooting_accuracy,
                             goals_to_shots,
                             total_shots]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Goals: {prediction[0]:.2f}")
