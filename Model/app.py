import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load trained model ---
model = pickle.load(open("Model/ridge_regression_model.pkl", "rb"))
scaler = pickle.load(open('Model/scaler.pkl', 'rb'))

st.title("Group 1-House Price Prediction Dashboard")


# --- Input fields ---
longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")

# Display map of the selected location
st.subheader("Property Location")
map_data = pd.DataFrame({
    'lon': [longitude],
    'lat': [latitude]
})
st.map(map_data, zoom=8)

# --- Predict ---
if st.button("Predict"):
    # Create input array with the EXACT same feature order as training
    X_new = np.array([[longitude, latitude, housing_median_age, 
                       total_rooms, total_bedrooms, population, 
                       households, median_income]])
    
    X_new_scaled = scaler.transform(X_new)

    prediction = model.predict(X_new_scaled)

    st.success(f"Predicted House Price: ${prediction[0]:.2f}")
