# Create a requirements.txt file for the project
requirements_content = """streamlit==1.28.0
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print("✅ Requirements file created!")
print("📁 File saved as: requirements.txt")
print("\n📦 Dependencies included:")
for line in requirements_content.strip().split('\n'):
    print(f"   - {line}")