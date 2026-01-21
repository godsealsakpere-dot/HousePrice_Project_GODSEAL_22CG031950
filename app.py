import streamlit as st
import joblib
import numpy as np
import os

# Page Configuration
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Custom CSS Loading
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("static/style.css")

# Title
st.title("🏠 House Price Prediction System")
st.write("Enter the details of the house below to estimate its sale price.")

# Load Model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'house_price_model.pkl')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please run Part A to generate 'house_price_model.pkl'.")
    st.stop()

# Input Form
st.subheader("Property Details")
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=6000, value=1000)

with col2:
    garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4])
    full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3])
    year_built = st.number_input("Year Built", min_value=1870, max_value=2026, value=2000)

# Predict Button
if st.button("Predict Price"):
    # Create input array matching the training order
    # ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
    input_data = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.success(f"Estimated House Price: ${prediction:,.2f}")