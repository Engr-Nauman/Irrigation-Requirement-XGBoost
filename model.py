import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Load Model and Encoders ---
model_dir = 'model_artifacts'

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(model_dir, 'xgboost_model.joblib'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    one_hot_encoder = joblib.load(os.path.join(model_dir, 'one_hot_encoder.joblib'))
    feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.joblib'))
    return model, label_encoder, one_hot_encoder, feature_columns

model, le, encoder, feature_columns = load_artifacts()

# --- 2. Define Input Fields (based on original df columns and transformations) ---
# These values should reflect the ranges/categories observed in your training data.
# For categorical features, we need the unique categories from the original 'df'
# To populate the select boxes, we'll use example categories here. In a real app,
# you'd ideally save these alongside your encoders.

# Numerical features ranges (example based on df.describe() or domain knowledge)
numerical_features_info = {
    'Soil_pH': {'min': 4.0, 'max': 9.0, 'default': 6.5},
    'Soil_Moisture': {'min': 0.0, 'max': 100.0, 'default': 50.0},
    'Organic_Carbon': {'min': 0.0, 'max': 5.0, 'default': 1.0},
    'Electrical_Conductivity': {'min': 0.0, 'max': 5.0, 'default': 2.0},
    'Temperature_C': {'min': 0.0, 'max': 40.0, 'default': 25.0},
    'Humidity': {'min': 0.0, 'max': 100.0, 'default': 70.0},
    'Rainfall_mm': {'min': 0.0, 'max': 300.0, 'default': 50.0},
    'Sunlight_Hours': {'min': 0.0, 'max': 12.0, 'default': 6.0},
    'Wind_Speed_kmh': {'min': 0.0, 'max': 50.0, 'default': 15.0},
    'Field_Area_hectare': {'min': 0.0, 'max': 10.0, 'default': 5.0},
    'Previous_Irrigation_mm': {'min': 0.0, 'max': 200.0, 'default': 30.0}
}

# Categorical features and their unique values (example, get from original df)
categorical_features_info = {
    'Soil_Type': ['Clay', 'Loamy', 'Sandy', 'Silt'],
    'Crop_Type': ['Wheat', 'Paddy', 'Maize', 'Sugarcane', 'Cotton'],
    'Crop_Growth_Stage': ['Germination', 'Vegetative', 'Flowering', 'Fruiting'],
    'Season': ['Summer', 'Monsoon', 'Winter'],
    'Irrigation_Type': ['Drip', 'Sprinkler', 'Surface', 'Subsurface'],
    'Water_Source': ['River', 'Well', 'Canal', 'Rainfed'],
    'Mulching_Used': ['Yes', 'No'],
    'Region': ['North', 'South', 'East', 'West']
}

# --- 3. Streamlit App Layout ---
st.set_page_config(page_title="Irrigation Need Prediction", layout="wide")
st.title("🌱 Irrigation Need Prediction App")
st.markdown("Enter the environmental and crop parameters to predict the irrigation need.")

# Create two columns for better layout
col1, col2 = st.columns(2)

user_input = {}

with col1:
    st.header("Environmental & Crop Parameters")
    for feature, info in numerical_features_info.items():
        user_input[feature] = st.slider(f"**{feature.replace('_', ' ')}**", 
                                        min_value=float(info['min']), 
                                        max_value=float(info['max']),
                                        value=float(info['default']),
                                        step=0.1)

with col2:
    st.header("Categorical Selections")
    for feature, options in categorical_features_info.items():
        user_input[feature] = st.selectbox(f"**{feature.replace('_', ' ')}**", options)

# --- 4. Prediction Logic ---
if st.button("Predict Irrigation Need"):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Separate numerical and categorical features
    input_numerical = input_df[list(numerical_features_info.keys())]
    input_categorical = input_df[list(categorical_features_info.keys())]

    # Apply OneHotEncoder to categorical features
    encoded_input_features = encoder.transform(input_categorical)
    encoded_input_df = pd.DataFrame(encoded_input_features, columns=encoder.get_feature_names_out(list(categorical_features_info.keys())))

    # Concatenate numerical and encoded categorical features
    processed_input = pd.concat([input_numerical, encoded_input_df], axis=1)

    # Ensure column order matches the training data
    # Important: This assumes 'feature_columns' saved from training matches the order expected.
    # If 'Mulching_Used_Yes' or 'Irrigation_Need_Medium' were dropped during training for df_final_reduced,
    # but not explicitly dropped in this Streamlit app's feature_columns logic, it needs careful handling.
    # For this current scenario, we did NOT drop columns, so feature_columns should match.
    
    # Reindex the processed input to match the feature_columns used during training
    processed_input = processed_input.reindex(columns=feature_columns, fill_value=0)

    # Make prediction
    prediction_encoded = model.predict(processed_input)
    prediction_label = le.inverse_transform(prediction_encoded)

    st.subheader("Prediction Result:")
    st.success(f"The predicted Irrigation Need is: **{prediction_label[0]}**")