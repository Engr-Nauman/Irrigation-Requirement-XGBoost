import streamlit as st
import pandas as pd
import joblib

# Set the page configuration
st.set_page_config(page_title="Irrigation Predictor", page_icon="💧", layout="centered")

# --- LOAD EXPORTED MODELS ---
@st.cache_resource
def load_models():
    model = joblib.load("irrigation_model.joblib")
    encoder = joblib.load("label_encoder.joblib")
    return model, encoder

model, label_encoder = load_models()

# --- APP HEADER ---
st.title("🌾 Irrigation Water Requirement Predictor 💧")
st.markdown("Predict the irrigation needs of your field based on soil, weather, crop, and field data! 🚀")

# --- INPUT FEATURES TABS ---
# Organizing the 19 input features into logical categories
tab1, tab2, tab3 = st.tabs(["🌱 Soil & Weather", "🚜 Crop & Field", "💦 Water Details"])

with tab1:
    st.header("Soil & Weather Parameters")
    soil_type = st.selectbox("Soil Type", ["Clay", "Silt", "Sandy"])
    soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.0, step=0.1)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, value=35.0, step=1.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=1.0, step=0.1)
    electrical_conductivity = st.number_input("Electrical Conductivity", min_value=0.0, value=1.5, step=0.1)
    
    st.divider()
    temperature_c = st.number_input("Temperature (°C)", value=25.0, step=1.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, value=1000.0, step=10.0)
    sunlight_hours = st.number_input("Sunlight Hours", min_value=0.0, value=8.0, step=0.5)
    wind_speed_kmh = st.number_input("Wind Speed (km/h)", min_value=0.0, value=10.0, step=1.0)

with tab2:
    st.header("Crop & Field Details")
    crop_type = st.selectbox("Crop Type", ["Wheat", "Maize", "Cotton"])
    crop_growth_stage = st.selectbox("Crop Growth Stage", ["Vegetative", "Flowering", "Harvest", "Sowing"])
    season = st.selectbox("Season", ["Rabi", "Zaid", "Kharif"])
    region = st.selectbox("Region", ["South", "Central", "North"])
    field_area_hectare = st.number_input("Field Area (hectares)", min_value=0.1, value=5.0, step=0.1)
    mulching_used = st.selectbox("Mulching Used", ["Yes", "No"])

with tab3:
    st.header("Water & Irrigation Details")
    irrigation_type = st.selectbox("Irrigation Type", ["Rainfed", "Canal", "Drip"])
    water_source = st.selectbox("Water Source", ["Reservoir", "Groundwater", "River"])
    previous_irrigation_mm = st.number_input("Previous Irrigation (mm)", min_value=0.0, value=30.0, step=1.0)

# --- PREDICTION SECTION ---
st.divider()

if st.button("🔮 Predict Irrigation Need"):
    # 1. Compile inputs into a dictionary matching your training data columns
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
        'Region': [region]
    }
    
    # 2. Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Note: If your XGBoost model expects encoded categorical features (and wasn't trained 
    # using enable_categorical=True), you must apply the same OneHotEncoding or LabelEncoding 
    # to 'input_df' here before calling predict().
    
    try:
        # 3. Make Prediction
        prediction_encoded = model.predict(input_df)
        
        # 4. Decode the prediction
        prediction = label_encoder.inverse_transform(prediction_encoded)
        
        # 5. Display the result
        st.success(f"🌟 **Predicted Irrigation Need:** {prediction[0]} 🌟")
        
    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {e}")
        st.info("💡 Hint: Ensure categorical variables are encoded exactly as they were during training.")
