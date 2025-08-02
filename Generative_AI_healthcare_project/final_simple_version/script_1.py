# Create a comprehensive Streamlit medical chatbot application
streamlit_app_code = '''import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Medical Symptom Checker & Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #A23B72;
    margin-bottom: 1rem;
}
.prediction-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #2E86AB;
    margin: 1rem 0;
}
.recommendation-box {
    background-color: #f5f5dc;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #A23B72;
    margin: 1rem 0;
}
.warning-box {
    background-color: #ffe4e1;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ff6b6b;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}

@st.cache_data
def load_data():
    """Load the medical dataset"""
    try:
        df = pd.read_csv('medical_symptom_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'medical_symptom_dataset.csv' is in the same directory.")
        return None

@st.cache_resource
def train_models(df):
    """Train multiple ML models"""
    # Prepare features and target
    X = df.drop('disease', axis=1)
    y = df['disease']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Train models and calculate accuracies
    trained_models = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        accuracies[name] = accuracy
    
    return trained_models, accuracies, X.columns.tolist()

def get_disease_recommendations(disease):
    """Get recommendations based on predicted disease"""
    recommendations = {
        'Common Cold': {
            'description': 'A viral infection of the upper respiratory tract.',
            'recommendations': [
                'Get plenty of rest',
                'Stay hydrated with fluids',
                'Use throat lozenges for sore throat',
                'Consider over-the-counter pain relievers',
                'Use a humidifier or breathe steam'
            ],
            'precautions': [
                'Wash hands frequently',
                'Avoid close contact with others',
                'Cover coughs and sneezes',
                'Stay home while symptomatic'
            ],
            'when_to_see_doctor': 'If symptoms worsen or last more than 10 days'
        },
        'Flu': {
            'description': 'A viral infection that attacks your respiratory system.',
            'recommendations': [
                'Rest and sleep as much as possible',
                'Drink plenty of fluids',
                'Consider antiviral medications if within 48 hours',
                'Use over-the-counter fever reducers',
                'Gargle with salt water'
            ],
            'precautions': [
                'Get annual flu vaccination',
                'Avoid close contact with sick people',
                'Practice good hygiene',
                'Stay home when sick'
            ],
            'when_to_see_doctor': 'If you have trouble breathing, chest pain, or high fever'
        },
        'COVID-19': {
            'description': 'A respiratory illness caused by the SARS-CoV-2 virus.',
            'recommendations': [
                'Isolate immediately',
                'Monitor symptoms closely',
                'Stay hydrated and rest',
                'Consider telehealth consultation',
                'Follow CDC guidelines'
            ],
            'precautions': [
                'Wear masks in public',
                'Practice social distancing',
                'Get vaccinated and boosted',
                'Improve indoor ventilation'
            ],
            'when_to_see_doctor': 'If you have difficulty breathing, chest pain, or severe symptoms'
        },
        'Migraine': {
            'description': 'A neurological condition characterized by severe headaches.',
            'recommendations': [
                'Rest in a dark, quiet room',
                'Apply cold or warm compress',
                'Stay hydrated',
                'Consider over-the-counter pain relievers',
                'Practice relaxation techniques'
            ],
            'precautions': [
                'Identify and avoid triggers',
                'Maintain regular sleep schedule',
                'Manage stress levels',
                'Stay hydrated'
            ],
            'when_to_see_doctor': 'If headaches are severe, frequent, or accompanied by neurological symptoms'
        },
        'Hypertension': {
            'description': 'High blood pressure that can lead to serious health complications.',
            'recommendations': [
                'Monitor blood pressure regularly',
                'Follow prescribed medications',
                'Maintain healthy diet (low sodium)',
                'Exercise regularly',
                'Manage stress levels'
            ],
            'precautions': [
                'Limit alcohol consumption',
                'Quit smoking',
                'Maintain healthy weight',
                'Regular medical check-ups'
            ],
            'when_to_see_doctor': 'For regular monitoring and if experiencing severe symptoms'
        },
        # Add more diseases as needed...
    }
    
    # Default recommendation if disease not in dictionary
    default = {
        'description': 'Please consult with a healthcare professional for proper diagnosis.',
        'recommendations': [
            'Monitor your symptoms',
            'Rest and stay hydrated',
            'Seek medical attention if symptoms worsen',
            'Follow general health guidelines'
        ],
        'precautions': [
            'Practice good hygiene',
            'Maintain healthy lifestyle',
            'Follow medical advice'
        ],
        'when_to_see_doctor': 'If symptoms persist or worsen'
    }
    
    return recommendations.get(disease, default)

def main():
    st.markdown('<h1 class="main-header">🏥 Medical Symptom Checker & Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Symptom Checker", "Dataset Info", "Model Performance", "About"])
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Train models if not already trained
    if not st.session_state.model_trained:
        with st.spinner("Training machine learning models..."):
            models, accuracies, feature_names = train_models(df)
            st.session_state.models = models
            st.session_state.accuracies = accuracies
            st.session_state.feature_names = feature_names
            st.session_state.model_trained = True
    
    if page == "Symptom Checker":
        st.markdown('<h2 class="sub-header">🔍 Symptom Checker</h2>', unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown('''
        <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult with a qualified healthcare provider for proper diagnosis and treatment.
        </div>
        ''', unsafe_allow_html=True)
        
        # Symptom selection
        st.subheader("Select your symptoms:")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        symptoms = st.session_state.feature_names
        selected_symptoms = {}
        
        # Distribute symptoms across columns
        symptoms_per_col = len(symptoms) // 3
        
        with col1:
            for i in range(0, symptoms_per_col):
                if i < len(symptoms):
                    selected_symptoms[symptoms[i]] = st.checkbox(symptoms[i].replace('_', ' ').title())
        
        with col2:
            for i in range(symptoms_per_col, 2 * symptoms_per_col):
                if i < len(symptoms):
                    selected_symptoms[symptoms[i]] = st.checkbox(symptoms[i].replace('_', ' ').title())
        
        with col3:
            for i in range(2 * symptoms_per_col, len(symptoms)):
                selected_symptoms[symptoms[i]] = st.checkbox(symptoms[i].replace('_', ' ').title())
        
        # Model selection
        st.subheader("Choose prediction model:")
        selected_model = st.selectbox("Select Model", list(st.session_state.models.keys()))
        
        if st.button("🔮 Predict Disease", type="primary"):
            # Check if any symptoms are selected
            if not any(selected_symptoms.values()):
                st.warning("Please select at least one symptom.")
                return
            
            # Create feature vector
            feature_vector = [1 if selected_symptoms.get(symptom, False) else 0 for symptom in symptoms]
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Make prediction
            model = st.session_state.models[selected_model]
            prediction = model.predict(feature_array)[0]
            probabilities = model.predict_proba(feature_array)[0] if hasattr(model, 'predict_proba') else None
            
            # Display prediction
            st.markdown(f'''
            <div class="prediction-box">
            <h3>🎯 Prediction Result</h3>
            <p><strong>Most likely condition:</strong> {prediction}</p>
            <p><strong>Model used:</strong> {selected_model}</p>
            <p><strong>Model accuracy:</strong> {st.session_state.accuracies[selected_model]:.2%}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Show probability distribution if available
            if probabilities is not None:
                prob_df = pd.DataFrame({
                    'Disease': model.classes_,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False).head(5)
                
                st.subheader("Top 5 Most Likely Conditions:")
                st.dataframe(prob_df, use_container_width=True)
            
            # Get and display recommendations
            recommendations = get_disease_recommendations(prediction)
            
            st.markdown(f'''
            <div class="recommendation-box">
            <h3>📋 Medical Information & Recommendations</h3>
            <p><strong>Description:</strong> {recommendations['description']}</p>
            
            <h4>💊 Recommended Actions:</h4>
            <ul>
            {"".join([f"<li>{rec}</li>" for rec in recommendations['recommendations']])}
            </ul>
            
            <h4>🛡️ Precautions:</h4>
            <ul>
            {"".join([f"<li>{prec}</li>" for prec in recommendations['precautions']])}
            </ul>
            
            <h4>🚨 When to See a Doctor:</h4>
            <p>{recommendations['when_to_see_doctor']}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    elif page == "Dataset Info":
        st.markdown('<h2 class="sub-header">📊 Dataset Information</h2>', unsafe_allow_html=True)
        
        st.write(f"**Dataset Shape:** {df.shape}")
        st.write(f"**Number of Diseases:** {df['disease'].nunique()}")
        st.write(f"**Number of Symptoms:** {len(df.columns) - 1}")
        
        st.subheader("Disease Distribution:")
        disease_counts = df['disease'].value_counts()
        st.bar_chart(disease_counts)
        
        st.subheader("Sample Data:")
        st.dataframe(df.head(10), use_container_width=True)
    
    elif page == "Model Performance":
        st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
        
        # Display model accuracies
        st.subheader("Model Accuracy Comparison:")
        accuracy_df = pd.DataFrame({
            'Model': list(st.session_state.accuracies.keys()),
            'Accuracy': list(st.session_state.accuracies.values())
        })
        
        st.bar_chart(accuracy_df.set_index('Model'))
        st.dataframe(accuracy_df, use_container_width=True)
        
        # Additional performance metrics could be added here
        st.info("💡 Higher accuracy indicates better model performance. Consider the trade-offs between different models based on your specific use case.")
    
    elif page == "About":
        st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Purpose
        This Medical Symptom Checker is designed to help users get preliminary insights into potential health conditions based on their symptoms. 
        
        ### 🛠️ Technology Stack
        - **Frontend:** Streamlit
        - **Machine Learning:** Scikit-learn
        - **Models Used:** Random Forest, Naive Bayes, Support Vector Machine
        - **Data Processing:** Pandas, NumPy
        
        ### 🔬 How It Works
        1. **Data Collection:** Uses a comprehensive symptom-disease dataset
        2. **Model Training:** Trains multiple ML models on the dataset
        3. **Prediction:** Takes user-selected symptoms and predicts likely conditions
        4. **Recommendations:** Provides evidence-based recommendations and precautions
        
        ### ⚠️ Important Disclaimers
        - This tool is for educational and informational purposes only
        - It should never replace professional medical advice
        - Always consult with qualified healthcare providers for proper diagnosis
        - In case of medical emergencies, seek immediate medical attention
        
        ### 🚀 Future Enhancements
        - Integration with medical APIs for updated information
        - More sophisticated NLP for symptom input
        - Integration with wearable device data
        - Multi-language support
        - Telemedicine integration
        
        ---
        **Developed with ❤️ for healthcare accessibility**
        """)

if __name__ == "__main__":
    main()
'''

# Save the Streamlit application
with open('medical_chatbot_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("✅ Streamlit Medical Chatbot Application Created!")
print("📁 File saved as: medical_chatbot_app.py")
print("\n🚀 To run the application:")
print("   streamlit run medical_chatbot_app.py")
print("\n📋 Features included:")
print("   - Symptom-based disease prediction")
print("   - Multiple ML models (Random Forest, Naive Bayes, SVM)")
print("   - Medical recommendations and precautions")
print("   - Model performance comparison")
print("   - Professional medical disclaimers")
print("   - User-friendly interface with custom styling")