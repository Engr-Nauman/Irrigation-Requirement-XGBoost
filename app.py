import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Irrigation Need Prediction App 💧",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Load Model Artifacts ---
# Define the directory where artifacts are saved
model_dir = 'model_artifacts'

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load(os.path.join(model_dir, 'xgboost_model.joblib'))
        le = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
        encoder = joblib.load(os.path.join(model_dir, 'one_hot_encoder.joblib'))
        feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.joblib'))
        categorical_features_for_ohe = joblib.load(os.path.join(model_dir, 'categorical_features_for_ohe.joblib'))
        return model, le, encoder, feature_columns, categorical_features_for_ohe
    except FileNotFoundError:
        st.error(f"Model artifacts not found. Please ensure 'model_artifacts' directory and its contents are in the same directory as this app. Looking for: {os.listdir(model_dir) if os.path.exists(model_dir) else 'directory not found'}")
        st.stop()

model, le, encoder, feature_columns, categorical_features_for_ohe = load_model_artifacts()

# --- Title and Description ---
st.title("Irrigation Need Prediction 💧")
st.markdown("This app predicts the irrigation need (High, Medium, Low) for a crop based on various environmental and soil parameters.")
st.markdown("--- ")

# --- Input Features ---
st.header("Environmental & Soil Parameters")

col1, col2 = st.columns(2)
with col1:
    soil_ph = st.slider("Soil pH", min_value=3.0, max_value=9.0, value=6.5, step=0.1)
    soil_moisture = st.slider("Soil Moisture (%) ", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    organic_carbon = st.slider("Organic Carbon (%) ", min_value=0.0, max_value=5.0, value=1.0, step=0.01)
    electrical_conductivity = st.slider("Electrical Conductivity (dS/m)", min_value=0.0, max_value=10.0, value=2.0, step=0.01)
    temperature_c = st.slider("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)

with col2:
    rainfall_mm = st.slider("Rainfall (mm)", min_value=0.0, max_value=300.0, value=50.0, step=0.1)
    sunlight_hours = st.slider("Sunlight Hours", min_value=0.0, max_value=15.0, value=8.0, step=0.1)
    wind_speed_kmh = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    field_area_hectare = st.slider("Field Area (hectare)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    previous_irrigation_mm = st.slider("Previous Irrigation (mm)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)


st.markdown("--- ")
st.header("Crop & Management Details")

col3, col4 = st.columns(2)
with col3:
    soil_type = st.selectbox("Soil Type", ['Clay', 'Loamy', 'Sandy', 'Silt'])
    crop_type = st.selectbox("Crop Type", ['Wheat', 'Maize', 'Paddy', 'Sugarcane', 'Cotton', 'Barley'])
    crop_growth_stage = st.selectbox("Crop Growth Stage", ['Vegetative', 'Flowering', 'Fruiting'])
    season = st.selectbox("Season", ['Monsoon', 'Winter', 'Summer'])

with col4:
    irrigation_type = st.selectbox("Irrigation Type", ['Drip', 'Sprinkler', 'Flood'])
    water_source = st.selectbox("Water Source", ['River', 'Well', 'Canal'])
    mulching_used = st.selectbox("Mulching Used", ['Yes', 'No'])
    region = st.selectbox("Region", ['North', 'South', 'East', 'West'])

# --- Create DataFrame for Prediction ---
input_data = {
    'Soil_Type': [soil_type],
    'Soil_pH': [soil_ph],
    'Soil_Moisture': [soil_moisture],
    'Organic_Carbon': [organic_carbon],
    'Electrical_Conductivity': [electrical_conductivity],
    'Temperature_C': [temperature_c],
    'Humidity': [humidity],
    'Rainfall_mm': [rainfall_mm],
    'Sunlight_Hours': [sunlight_hours],
    'Wind_Speed_kmh': [wind_speed_kmh],
    'Crop_Type': [crop_type],
    'Crop_Growth_Stage': [crop_growth_stage],
    'Season': [season],
    'Irrigation_Type': [irrigation_type],
    'Water_Source': [water_source],
    'Field_Area_hectare': [field_area_hectare],
    'Mulching_Used': [mulching_used],
    'Previous_Irrigation_mm': [previous_irrigation_mm],
    'Region': [region],
}

input_df = pd.DataFrame(input_data)

# --- Preprocessing Input Data ---
# Separate numerical and categorical columns from the input
numerical_cols_input = input_df.select_dtypes(include=np.number).columns
# Use the saved list of categorical features for OHE
categorical_cols_to_encode = [col for col in categorical_features_for_ohe if col in input_df.columns]

# Apply OneHotEncoder to categorical features
encoded_features_input = encoder.transform(input_df[categorical_cols_to_encode])
encoded_df_input = pd.DataFrame(encoded_features_input, columns=encoder.get_feature_names_out(categorical_cols_to_encode))

# Drop original categorical columns and concatenate with encoded ones
input_processed = input_df[numerical_cols_input].reset_index(drop=True)
input_processed = pd.concat([input_processed, encoded_df_input], axis=1)

# Ensure all columns from training are present and in the correct order
# Add missing columns with 0, drop extra columns
final_input = pd.DataFrame(0, index=[0], columns=feature_columns) # Initialize with all expected columns and 0s
for col in input_processed.columns:
    if col in final_input.columns:
        final_input[col] = input_processed[col].iloc[0] # Populate with user input


# --- Prediction Button ---
st.markdown("--- ")
if st.button("Predict Irrigation Need 🌱"): 
    prediction_encoded = model.predict(final_input)
    prediction_label = le.inverse_transform(prediction_encoded)
    
    st.success(f"Predicted Irrigation Need: **{prediction_label[0]}**")
    st.balloons()


