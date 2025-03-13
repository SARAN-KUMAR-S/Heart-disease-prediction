import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_load_success = True
except Exception as e:
    model_load_success = False
    error_message = str(e)

# Get feature names from your dataset
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Helper function for prediction
def predict_heart_disease(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)
    return prediction[0], prediction_proba[0]

# App title and description
st.title("❤️ Heart Disease Prediction Tool")
st.markdown("""
This application uses machine learning to predict the likelihood of heart disease based on various health metrics.
Enter your information below to get a prediction.
""")

# Create sidebar for inputs
st.sidebar.header("Patient Information")

# Description dictionaries for feature explanations
sex_desc = {0: "Female", 1: "Male"}
cp_desc = {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}
fbs_desc = {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"}
restecg_desc = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
exang_desc = {0: "No", 1: "Yes"}
slope_desc = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
thal_desc = {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}

# Input form
with st.sidebar.form("user_inputs"):
    # Demographic inputs
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", options=list(sex_desc.keys()), format_func=lambda x: sex_desc[x])
    
    # Clinical inputs
    st.markdown("### Clinical Measurements")
    cp = st.selectbox("Chest Pain Type", options=list(cp_desc.keys()), format_func=lambda x: cp_desc[x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar", options=list(fbs_desc.keys()), format_func=lambda x: fbs_desc[x])
    
    # ECG results
    st.markdown("### ECG Results")
    restecg = st.selectbox("Resting ECG Results", options=list(restecg_desc.keys()), format_func=lambda x: restecg_desc[x])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=list(exang_desc.keys()), format_func=lambda x: exang_desc[x])
    
    # ST depression and slope
    st.markdown("### Exercise Test Results")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=list(slope_desc.keys()), format_func=lambda x: slope_desc[x])
    
    # Angiographic findings
    st.markdown("### Angiographic Findings")
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia", options=list(thal_desc.keys()), format_func=lambda x: thal_desc[x])
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# Main content area
if not model_load_success:
    st.error(f"Failed to load model: {error_message}")
    st.info("Please make sure the model files 'heart_disease_xgb_model.pkl' and 'scaler.pkl' are in the same directory as this script.")
else:
    # Process prediction when form is submitted
    if submitted:
        # Collect all input features
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        # Make prediction
        prediction, prediction_proba = predict_heart_disease(features)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Prediction Result")
            if prediction == 1:
                st.error("⚠️ Heart Disease Detected")
                st.metric("Risk Level", f"High ({prediction_proba[1]:.2%} probability)")
            else:
                st.success("✅ No Heart Disease Detected")
                st.metric("Risk Level", f"Low ({prediction_proba[0]:.2%} probability)")
            
            # Display prediction probability as a gauge
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.barh([0], [prediction_proba[0]], color='green', alpha=0.6)
            ax.barh([0], [prediction_proba[1]], left=[prediction_proba[0]], color='red', alpha=0.6)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlabel('Probability')
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            st.header("Patient Profile")
            
            # Display input values in a more readable format
            profile_data = {
                "Age": age,
                "Sex": sex_desc[sex],
                "Chest Pain Type": cp_desc[cp],
                "Resting Blood Pressure": f"{trestbps} mm Hg",
                "Cholesterol": f"{chol} mg/dl",
                "Fasting Blood Sugar > 120 mg/dl": "Yes" if fbs == 1 else "No",
                "Resting ECG": restecg_desc[restecg],
                "Max Heart Rate": thalach,
                "Exercise Induced Angina": "Yes" if exang == 1 else "No",
                "ST Depression": oldpeak,
                "ST Segment Slope": slope_desc[slope],
                "Major Vessels (0-4)": ca,
                "Thalassemia": thal_desc[thal]
            }
            
            # Convert to DataFrame for display
            profile_df = pd.DataFrame(list(profile_data.items()), columns=["Metric", "Value"])
            st.table(profile_df)
        
        # Additional information
        st.subheader("Understanding Your Results")
        st.markdown("""
        This prediction is based on statistical patterns found in historical heart disease data and should be used for informational purposes only.
        
        **Important:** This tool is not a substitute for professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.
        """)
    else:
        # Display info when app first loads
        st.info("👈 Fill out the patient information in the sidebar and click 'Predict' to get a heart disease risk assessment.")
        
        # Display feature importance if available
        try:
            st.subheader("Model Feature Importance")
            st.markdown("This chart shows which factors have the most influence on heart disease prediction in our model:")
            
            # Create feature importance chart
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax)
            ax.set_title("Feature Importance in Heart Disease Prediction")
            ax.set_xlabel("Relative Importance")
            st.pyplot(fig)
            
            # Explanation of key features
            st.subheader("Key Risk Factors")
            st.markdown("""
            - **Chest Pain Type**: Different types of chest pain can indicate different levels of heart disease risk
            - **Maximum Heart Rate**: How your heart performs during exercise can be indicative of heart health
            - **Number of Major Vessels**: More blocked vessels typically indicate higher risk
            - **ST Depression**: ST segment changes during exercise testing are important diagnostic indicators
            """)
        except:
            st.write("Feature importance visualization not available.")

# Add a footer
st.markdown("---")
st.markdown("❤️ Heart Disease Prediction Tool | Guided by RAGHUNATHAN ")