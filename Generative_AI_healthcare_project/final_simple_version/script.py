# Let's create a sample dataset to demonstrate the medical chatbot project structure
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create a comprehensive medical dataset
diseases = [
    'Common Cold', 'Flu', 'COVID-19', 'Migraine', 'Hypertension', 
    'Diabetes', 'Asthma', 'Bronchitis', 'Pneumonia', 'Gastritis',
    'Food Poisoning', 'Appendicitis', 'Kidney Stones', 'UTI', 'Anemia',
    'Allergic Rhinitis', 'Sinusitis', 'Tonsillitis', 'Laryngitis', 'Conjunctivitis'
]

symptoms = [
    'fever', 'cough', 'headache', 'fatigue', 'sore_throat', 'runny_nose',
    'body_aches', 'nausea', 'vomiting', 'diarrhea', 'stomach_pain', 'chest_pain',
    'shortness_of_breath', 'dizziness', 'loss_of_appetite', 'chills', 'sweating',
    'muscle_pain', 'joint_pain', 'skin_rash', 'itchy_eyes', 'sneezing',
    'difficulty_swallowing', 'hoarse_voice', 'red_eyes', 'frequent_urination',
    'burning_urination', 'back_pain', 'weight_loss', 'weight_gain', 'blurred_vision'
]

recommendations = [
    'Rest and hydration', 'Over-the-counter pain relievers', 'Antiviral medication',
    'Antibiotics', 'Steam inhalation', 'Throat lozenges', 'Increased fluid intake',
    'Warm salt water gargle', 'Cold compress', 'Avoid allergens', 'Nasal decongestants',
    'Cough suppressants', 'Bronchodilators', 'Pain management', 'Dietary changes',
    'Regular exercise', 'Blood pressure monitoring', 'Blood glucose monitoring',
    'Seek immediate medical attention', 'Follow-up with healthcare provider'
]

# Create a more realistic symptom-disease association matrix
disease_symptom_map = {
    'Common Cold': ['runny_nose', 'sore_throat', 'cough', 'sneezing', 'fatigue'],
    'Flu': ['fever', 'body_aches', 'fatigue', 'cough', 'headache', 'chills'],
    'COVID-19': ['fever', 'cough', 'shortness_of_breath', 'fatigue', 'loss_of_appetite'],
    'Migraine': ['headache', 'nausea', 'dizziness', 'blurred_vision'],
    'Hypertension': ['headache', 'dizziness', 'chest_pain', 'shortness_of_breath'],
    'Diabetes': ['frequent_urination', 'weight_loss', 'fatigue', 'blurred_vision'],
    'Asthma': ['shortness_of_breath', 'cough', 'chest_pain', 'wheezing'],
    'Bronchitis': ['cough', 'chest_pain', 'fatigue', 'shortness_of_breath'],
    'Pneumonia': ['fever', 'cough', 'chest_pain', 'shortness_of_breath', 'chills'],
    'Gastritis': ['stomach_pain', 'nausea', 'vomiting', 'loss_of_appetite'],
    'Food Poisoning': ['nausea', 'vomiting', 'diarrhea', 'stomach_pain', 'fever'],
    'Appendicitis': ['stomach_pain', 'nausea', 'vomiting', 'fever'],
    'Kidney Stones': ['back_pain', 'frequent_urination', 'burning_urination', 'nausea'],
    'UTI': ['burning_urination', 'frequent_urination', 'back_pain', 'fever'],
    'Anemia': ['fatigue', 'dizziness', 'weight_loss', 'shortness_of_breath'],
    'Allergic Rhinitis': ['runny_nose', 'sneezing', 'itchy_eyes', 'sore_throat'],
    'Sinusitis': ['headache', 'runny_nose', 'sore_throat', 'fatigue'],
    'Tonsillitis': ['sore_throat', 'difficulty_swallowing', 'fever', 'headache'],
    'Laryngitis': ['hoarse_voice', 'sore_throat', 'cough', 'difficulty_swallowing'],
    'Conjunctivitis': ['red_eyes', 'itchy_eyes', 'runny_nose']
}

# Generate the dataset
data = []
for i in range(2000):  # Generate 2000 samples
    disease = random.choice(diseases)
    
    # Create symptom vector
    symptom_vector = {symptom: 0 for symptom in symptoms}
    
    # Add primary symptoms for the disease
    if disease in disease_symptom_map:
        primary_symptoms = disease_symptom_map[disease]
        for symptom in primary_symptoms:
            if symptom in symptom_vector:
                symptom_vector[symptom] = 1
        
        # Add some random additional symptoms (noise)
        num_additional = random.randint(0, 3)
        additional_symptoms = random.sample([s for s in symptoms if s not in primary_symptoms], 
                                          min(num_additional, len(symptoms) - len(primary_symptoms)))
        for symptom in additional_symptoms:
            if random.random() < 0.3:  # 30% chance to add noise
                symptom_vector[symptom] = 1
    
    # Create row
    row = {'disease': disease}
    row.update(symptom_vector)
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Display basic information about the dataset
print("Medical Symptom-Disease Dataset Created")
print(f"Dataset shape: {df.shape}")
print(f"Number of diseases: {df['disease'].nunique()}")
print(f"Number of symptoms: {len(symptoms)}")
print("\nFirst few rows:")
print(df.head())

# Save the dataset
df.to_csv('medical_symptom_dataset.csv', index=False)
print("\nDataset saved as 'medical_symptom_dataset.csv'")

# Display disease distribution
print("\nDisease distribution:")
print(df['disease'].value_counts().head(10))