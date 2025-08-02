#!/usr/bin/env python3
"""
Medical Chatbot Project Setup Script
Automates the installation and setup of the medical symptom checker application.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing packages. Please install manually using:")
        print("   pip install -r requirements.txt")
        return False

def check_files():
    """Check if all required files are present"""
    required_files = [
        "medical_chatbot_app.py",
        "medical_symptom_dataset.csv", 
        "requirements.txt",
        "README.md"
    ]

    print("🔍 Checking for required files...")
    missing_files = []

    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            missing_files.append(file)

    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All required files found!")
        return True

def run_application():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Medical Chatbot Application...")
    print("📌 The application will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("\n⚠️ To stop the application, press Ctrl+C in this terminal")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "medical_chatbot_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it using:")
        print("   pip install streamlit")

def main():
    print("🏥 Medical Symptom Checker & Recommendation System")
    print("=" * 55)
    print("Setting up the application...\n")

    # Check for required files
    if not check_files():
        print("\n❌ Setup failed. Please ensure all files are in the correct location.")
        return

    # Install requirements
    if not install_requirements():
        response = input("\nWould you like to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Ask user if they want to run the application
    print("\n" + "=" * 55)
    response = input("🚀 Would you like to launch the application now? (y/n): ")

    if response.lower() == 'y':
        run_application()
    else:
        print("\n📝 To run the application later, use:")
        print("   streamlit run medical_chatbot_app.py")
        print("\n📖 Check README.md for detailed instructions")

if __name__ == "__main__":
    main()
