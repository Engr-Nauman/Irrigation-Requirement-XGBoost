# 💧 Irrigation Need Prediction

This repository hosts a machine learning project aimed at predicting the irrigation needs of crops based on various environmental and agricultural parameters. The project leverages an XGBoost classifier to classify irrigation needs into 'High', 'Medium', or 'Low' categories.

## Project Goal

The primary goal of this project is to develop a predictive model that can assist farmers and agricultural stakeholders in optimizing water usage for irrigation, thereby promoting sustainable farming practices and improving crop yield.

## Code Overview

The `irrigation_prediction_notebook.ipynb` Colab notebook contains the full workflow:

1.  **Data Loading**: The dataset is loaded from a specified source (e.g., Kaggle).
2.  **Exploratory Data Analysis (EDA)**: Initial checks for data shape, types, and missing values are performed. Correlation analysis is conducted for numerical features.
3.  **Feature Engineering**: Categorical features are transformed using One-Hot Encoding to prepare them for machine learning algorithms.
4.  **Model Training**: An XGBoost Classifier is trained on the processed data. The target variable `Irrigation_Need` is reconstructed from its one-hot encoded components and then LabelEncoded.
5.  **Model Evaluation**: The performance of the trained model is evaluated using accuracy and a detailed classification report.
6.  **Model Export**: The trained XGBoost model, LabelEncoder, OneHotEncoder, and the list of feature columns are saved to disk for future deployment.

## Model and Exported Artifacts

After training, the following model artifacts are saved in the `model_artifacts/` directory:

*   `xgboost_model.joblib`: The trained XGBoost Classifier model.
*   `label_encoder.joblib`: The `LabelEncoder` object used to transform target labels (High, Low, Medium) into numerical format and vice-versa.
*   `one_hot_encoder.joblib`: The `OneHotEncoder` object used to transform categorical input features into numerical format.
*   `feature_columns.joblib`: A list of column names representing the exact order and set of features expected by the trained model. This is crucial for consistent input during prediction.

These artifacts enable the deployment of the model without retraining and ensure that new input data is processed identically to the training data.

## Streamlit Application (`app.py`)

This repository also includes a Streamlit application (`model.py`) that provides an interactive web interface for making predictions using the exported model. The app allows users to input various environmental and crop parameters via sliders and select boxes, and then instantly get a prediction for the irrigation need.

### How to Run the Streamlit App Locally

1.  **Clone this repository** to your local machine.
2.  **Navigate to the repository directory** in your terminal.
3.  **Ensure you have Python installed** (preferably Python 3.8+).
4.  **Install the required libraries**: You can find a `requirements.txt` file (if provided) or install them manually:
    ```bash
    pip install streamlit pandas scikit-learn xgboost joblib
    ```
5.  **Run the Streamlit app**: Make sure the `model_artifacts` folder is in the same directory as `model.py`.
    ```bash
    streamlit run model.py
    ```

    This command will open the Streamlit application in your default web browser.

### Deployment to Streamlit Community Cloud

You can easily deploy this Streamlit application to the [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting it to this GitHub
repository. Ensure that:

*   Your `app.py` is in the root directory or a clearly specified sub-directory.
*   The `model_artifacts` folder and its contents are committed to the repository.
*   A `requirements.txt` file listing all dependencies (streamlit, pandas, scikit-learn, xgboost, joblib) is present in the repository.

### Demo App
https://irrigation-requirement-xgboost-en.streamlit.app/
Click on the link to test the Demo App! Cheers

#### Development Credits
Armaan
