###############################################
# Streamlit App: Taxi Fare Prediction
###############################################

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# 1️⃣ Load the trained model
model_path = r"C:\Users\nancy\Music\fare_prediction_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# 2️⃣ Streamlit UI
st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚖")
st.title("🚖 Taxi Fare Prediction App")

st.markdown("""
This app predicts the taxi fare based on trip details.
Enter trip details below:
""")

# 3️⃣ Input fields
trip_distance = st.number_input("Trip Distance (km):", min_value=0.1, max_value=100.0, value=5.0)
pickup_hour = st.slider("Pickup Hour (0-23):", min_value=0, max_value=23, value=12)
pickup_day = st.selectbox("Pickup Day:", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
is_night = st.checkbox("Is it Night (Before 6 AM or After 10 PM)?", value=False)

# Convert pickup_day to numeric as in training (Monday=0)
day_mapping = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
pickup_day_num = day_mapping[pickup_day]
is_night_num = 1 if is_night else 0

# 4️⃣ Prepare input dataframe to match model
input_data = pd.DataFrame([[trip_distance, pickup_hour, pickup_day_num, is_night_num]],
                          columns=model.feature_names_in_)

# 5️⃣ Predict
if st.button("Predict Fare"):
    predicted_fare = model.predict(input_data)
    st.success(f"Predicted Fare: ₹{predicted_fare[0]:.2f}")

# 6️⃣ Footer
st.markdown("---")
st.caption("Built by [Your Name]")
