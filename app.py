import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# ==========================================
# 1. LOAD EXPORTED ARTIFACTS
# ==========================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('irrigation_model.joblib')
    le = joblib.load('label_encoder.joblib')
    ohe = joblib.load('one_hot_encoder.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    return model, le, ohe, feature_columns

model, label_encoder, one_hot_encoder, feature_columns = load_artifacts()

# ==========================================
# 2. STREAMLIT UI & USER INPUTS
# ==========================================
st.title("Irrigation Water Requirement Prediction")
st.write("Enter the environmental and crop parameters below to predict irrigation needs.")

# Create columns for a cleaner UI layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Categorical Features")
    soil_type = st.selectbox("Soil Type", ["Clay", "Silt", "Sandy"])
    crop_type = st.selectbox("Crop Type", ["Wheat", "Maize", "Cotton"])
    growth_stage = st.selectbox("Crop Growth Stage", ["Sowing", "Vegetative", "Flowering", "Harvest"])
    season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
    irrigation_type = st.selectbox("Irrigation Type", ["Rainfed", "Canal", "Drip"])
    water_source = st.selectbox("Water Source", ["Reservoir", "Groundwater", "River"])
    region = st.selectbox("Region", ["North", "Central", "South"])
    mulching_used = st.selectbox("Mulching Used", ["Yes", "No"])

with col2:
    st.subheader("Numerical Features")
    soil_ph = st.number_input("Soil pH", value=6.5)
    soil_moisture = st.number_input("Soil Moisture (%)", value=30.0)
    organic_carbon = st.number_input("Organic Carbon", value=1.0)
    electrical_conductivity = st.number_input("Electrical Conductivity", value=1.5)
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    rainfall = st.number_input("Rainfall (mm)", value=500.0)
    sunlight = st.number_input("Sunlight Hours", value=8.0)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
    field_area = st.number_input("Field Area (hectare)", value=5.0)
    prev_irrigation = st.number_input("Previous Irrigation (mm)", value=20.0)

# ==========================================
# 3. PREDICTION PIPELINE
# ==========================================
if st.button("Predict Irrigation Need"):
    
    # 3.1. Create a DataFrame from the user inputs
    input_data = {
        'Soil_Type': [soil_type],
        'Soil_pH': [soil_ph],
        'Soil_Moisture': [soil_moisture],
        'Organic_Carbon': [organic_carbon],
        'Electrical_Conductivity': [electrical_conductivity],
        'Temperature_C': [temperature],
        'Humidity': [humidity],
        'Rainfall_mm': [rainfall],
        'Sunlight_Hours': [sunlight],
        'Wind_Speed_kmh': [wind_speed],
        'Crop_Type': [crop_type],
        'Crop_Growth_Stage': [growth_stage],
        'Season': [season],
        'Irrigation_Type': [irrigation_type],
        'Water_Source': [water_source],
        'Field_Area_hectare': [field_area],
        'Mulching_Used': [mulching_used],
        'Previous_Irrigation_mm': [prev_irrigation],
        'Region': [region]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Separate columns exactly as they were likely split during training[cite: 2]
    categorical_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
                        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
    numerical_cols = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
                      'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours', 
                      'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
    
    # 3.2. Apply One-Hot Encoding to categorical columns
    encoded_cats = one_hot_encoder.transform(input_df[categorical_cols])
    
    # Check if the output is a sparse matrix and convert to dense array if necessary
    if hasattr(encoded_cats, "toarray"):
        encoded_cats = encoded_cats.toarray()
        
    encoded_cats_df = pd.DataFrame(
        encoded_cats, 
        columns=one_hot_encoder.get_feature_names_out(categorical_cols)
    )
    
    # 3.3. Concatenate numerical and encoded categorical data
    num_df = input_df[numerical_cols].reset_index(drop=True)
    processed_df = pd.concat([num_df, encoded_cats_df], axis=1)
    
    # 3.4. FIX THE DMATRIX ERROR: Reindex the dataframe to match the training features
    # This ensures that all columns expected by XGBoost are present in the exact same order.
    # Any missing columns (e.g., a specific category not chosen) are filled with 0.
    final_df = processed_df.reindex(columns=feature_columns, fill_value=0)
    
    # 3.5. Make Prediction
    try:
        prediction_encoded = model.predict(final_df)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)
        
        st.success(f"**Predicted Irrigation Need:** {prediction_label[0]}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
