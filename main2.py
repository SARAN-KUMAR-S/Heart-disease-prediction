import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from groq import Groq

# Set page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Configure Groq client - improved with hidden API key management
def get_groq_client():
    # Get API key from environment variable
    #api_key = os.environ.get("GROQ_API_KEY")
    api_key = "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"
    
    if not api_key:
        return None, "API key not configured. Please contact the administrator."
    
    try:
        client = Groq(api_key=api_key)
        # Test the client with a simple request to verify it works
        test_response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.3-70b-versatile",
            max_tokens=10
        )
        # If we get here, the client is working
        return client, "success"
    except Exception as e:
        return None, f"Error initializing Groq client. Please contact the administrator."

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

# Helper function for healthcare chatbot
def get_health_advice(question):
    # Get fresh client each time to ensure we have the latest API key
    client, status = get_groq_client()
    
    if status != "success":
        return "Our healthcare assistant is currently unavailable. Please try again later."
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful healthcare assistant focused on heart health education.
                    Provide informative answers about heart disease, prevention, risk factors, and lifestyle changes.
                    Always remind users that you are not a substitute for professional medical advice.
                    Keep responses concise, factual, and evidence-based.
                    Do not diagnose conditions or prescribe treatments."""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return "Our healthcare assistant is currently unavailable. Please try again later."

# Create tabs for Prediction and Chatbot
tab1, tab2 = st.tabs(["❤️ Heart Disease Prediction", "💬 Healthcare Assistant"])

with tab1:
    # App title and description
    st.title("❤️ Heart Disease Prediction Tool")
    st.markdown("""
    This application uses machine learning to predict the likelihood of heart disease based on various health metrics.
    Enter your information in the sidebar to get a prediction.
    """)
    
    # Main content for prediction tool
    if not model_load_success:
        st.error(f"Failed to load model: {error_message}")
        st.info("Please make sure the model files 'heart_disease_xgb_model.pkl' and 'scaler.pkl' are in the same directory as this script.")
    else:
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
            
            # Add suggestion to try the chatbot
            st.info("💡 Have questions about your results? Try our Healthcare Assistant tab for more information about heart health.")
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

with tab2:
    st.title("💬 Healthcare Assistant")
    st.markdown("""
    Ask questions about heart health, risk factors, lifestyle changes, or understanding medical terms.
    Our AI assistant provides educational information to help you better understand heart disease.
    """)
    
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_question = st.chat_input("Ask a question about heart health...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get AI response with progress indication
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_health_advice(user_question)
            st.write(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Add some example questions to help users get started
    if not st.session_state.chat_history:
        st.markdown("### Example Questions:")
        example_questions = [
            "What are the main risk factors for heart disease?",
            "How can I improve my heart health through diet?",
            "What exercises are best for heart health?",
            "What does cholesterol have to do with heart disease?",
            "What is the difference between systolic and diastolic blood pressure?"
        ]
        
        # Display example questions as buttons
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_q_{i}"):
                    # Simulate clicking the chat input with this question
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    with st.chat_message("user"):
                        st.write(question)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = get_health_advice(question)
                        st.write(response)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()

# Add a footer
st.markdown("---")
st.markdown("❤️ Heart Disease Prediction Tool | Guided by RAGHUNATHAN")