# Medical Symptom Checker & Recommendation System 🏥

A comprehensive medical chatbot built with Streamlit that uses machine learning to predict diseases based on user symptoms and provide medical recommendations.

## 🌟 Features

- **Symptom-Based Disease Prediction**: Select symptoms and get AI-powered disease predictions
- **Multiple ML Models**: Choose from Random Forest, Naive Bayes, and SVM algorithms 
- **Medical Recommendations**: Get evidence-based treatment recommendations and precautions
- **Interactive Dashboard**: User-friendly interface with custom styling
- **Model Performance Comparison**: Compare accuracy across different ML models
- **Educational Tool**: Comprehensive medical information with proper disclaimers

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run medical_chatbot_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
medical-chatbot/
├── medical_chatbot_app.py          # Main Streamlit application
├── medical_symptom_dataset.csv     # Training dataset
├── requirements.txt                # Python dependencies
└── README.md                      # Project documentation
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Models**: Random Forest, Naive Bayes, Support Vector Machine

## 📊 Dataset Information

- **Size**: 2000 samples across 20 diseases
- **Features**: 31 different symptoms (binary encoded)
- **Diseases Covered**: Common Cold, Flu, COVID-19, Migraine, Hypertension, Diabetes, and more
- **Format**: CSV with symptom columns (0/1) and disease labels

## 🎯 How It Works

1. **Data Loading**: Loads the medical symptom dataset
2. **Model Training**: Trains multiple ML models (RF, NB, SVM)
3. **User Input**: User selects symptoms via checkboxes
4. **Prediction**: Selected model predicts the most likely disease
5. **Recommendations**: Provides medical information and recommendations
6. **Results**: Displays prediction confidence and top likely conditions

## 📈 Model Performance

The system compares three machine learning algorithms:

- **Random Forest**: Ensemble method, typically highest accuracy
- **Naive Bayes**: Probabilistic classifier, fast and efficient 
- **Support Vector Machine**: Robust classifier with probability estimates

## 🩺 Medical Recommendations

For each predicted condition, the system provides:

- **Disease Description**: Medical explanation of the condition
- **Treatment Recommendations**: Evidence-based suggested actions
- **Precautions**: Preventive measures and lifestyle advice
- **When to See a Doctor**: Clear guidance on seeking professional help

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This tool is for educational and informational purposes
- **Not Medical Advice**: Does not replace professional medical consultation
- **Emergency Situations**: Seek immediate medical attention for emergencies
- **Accuracy Limitations**: ML predictions may not always be accurate



## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes and improvements
- New features and enhancements
- Documentation updates
- Dataset expansions
- Model optimizations

## 📄 License

This project is for educational purposes. Please ensure compliance with medical data regulations in your jurisdiction.

## 🔧 Troubleshooting

### Common Issues:

1. **Dataset not found error**:
   - Ensure `medical_symptom_dataset.csv` is in the same directory
   - Run the dataset generation code if missing

2. **Import errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Verify Python version compatibility

3. **Streamlit issues**:
   - Update Streamlit: `pip install --upgrade streamlit`
   - Clear cache: Delete `.streamlit` folder



---

**⚠️ Medical Disclaimer**: This application is intended for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.


