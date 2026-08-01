import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Irrigation Need Predictor",
    page_icon="🌾",
    layout="wide"
)

# Load the trained Model and Label Encoder using Joblib
@st.cache_resource
def load_assets():
    model = joblib.load('irrigation_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    return model, label_encoder

try:
    model, label_encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure 'irrigation_model.joblib' and 'label_encoder.joblib' exist.")
    st.stop()

st.title("🌾 Irrigation Water Requirement Predictor")
st.markdown("Predict the **Irrigation Need** for crops based on soil, environmental, and agricultural metrics.")

st.divider()

# Organize Inputs into Tabs/Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header("🌱 Soil Features")
    soil_type = st.selectbox("Soil Type", ["Clay", "Silt", "Sandy"])
    soil_ph = st.slider("Soil pH", 4.0, 10.0, 6.5, step=0.01)
    soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 30.0, step=0.1)
    organic_carbon = st.number_input("Organic Carbon (%)", 0.0, 5.0, 0.8, step=0.01)
    electrical_cond = st.number_input("Electrical Conductivity (dS/m)", 0.0, 10.0, 1.5, step=0.01)

with col2:
    st.header("🌤️ Weather Metrics")
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 28.0, step=0.1)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 3000.0, 500.0, step=1.0)
    sunlight_hours = st.slider("Sunlight Hours", 0.0, 16.0, 8.0, step=0.1)
    wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 10.0, step=0.1)

with col3:
    st.header("🌽 Crop & Field Settings")
    crop_type = st.selectbox("Crop Type", ["Wheat", "Maize", "Cotton"])
    crop_stage = st.selectbox("Crop Growth Stage", ["Sowing", "Vegetative", "Flowering", "Harvest"])
    season = st.selectbox("Season", ["Rabi", "Kharif", "Zaid"])
    irrigation_type = st.selectbox("Irrigation Type", ["Rainfed", "Canal", "Drip"])
    water_source = st.selectbox("Water Source", ["Reservoir", "Groundwater", "River"])
    region = st.selectbox("Region", ["North", "South", "Central"])
    mulching = st.selectbox("Mulching Used", ["Yes", "No"])
    field_area = st.number_input("Field Area (Hectare)", 0.1, 100.0, 5.0, step=0.1)
    prev_irrigation = st.number_input("Previous Irrigation (mm)", 0.0, 200.0, 20.0, step=0.1)

# Prediction Logic
st.divider()

if st.button("🔮 Predict Irrigation Need", type="primary", use_container_width=True):
    # Create DataFrame from User Inputs matching training features
    input_data = pd.DataFrame([{
        'Soil_Type': soil_type,
        'Soil_pH': soil_ph,
        'Soil_Moisture': soil_moisture,
        'Organic_Carbon': organic_carbon,
        'Electrical_Conductivity': electrical_cond,
        'Temperature_C': temperature,
        'Humidity': humidity,
        'Rainfall_mm': rainfall,
        'Sunlight_Hours': sunlight_hours,
        'Wind_Speed_kmh': wind_speed,
        'Crop_Type': crop_type,
        'Crop_Growth_Stage': crop_stage,
        'Season': season,
        'Irrigation_Type': irrigation_type,
        'Water_Source': water_source,
        'Field_Area_hectare': field_area,
        'Mulching_Used': mulching,
        'Previous_Irrigation_mm': prev_irrigation,
        'Region': region
    }])
    
    # Perform Inference
    prediction_encoded = model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    probabilities = model.predict_proba(input_data)[0]

    # Display Results
    st.subheader("Prediction Result")
    
    if prediction_label == "Low":
        st.success(f"**Predicted Irrigation Need:** {prediction_label}")
    elif prediction_label == "Medium":
        st.warning(f"**Predicted Irrigation Need:** {prediction_label}")
    else:
        st.error(f"**Predicted Irrigation Need:** {prediction_label}")

    # Show Prediction Probabilities
    classes = label_encoder.classes_
    prob_df = pd.DataFrame({'Irrigation Need': classes, 'Probability': probabilities})
    st.bar_chart(prob_df.set_index('Irrigation Need'))