#!/usr/bin/env python3
"""
Test script for the Credit Risk Prediction Web App
This script tests both the web interface and API endpoints
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
TEST_DATA = {
    "person_age": 30,
    "person_income": 50000,
    "person_emp_length": 5,
    "person_home_ownership": "RENT",
    "loan_amnt": 10000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.2,
    "loan_grade": "B",
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 8
}

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed:")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Timestamp: {data['timestamp']}")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_api_prediction():
    """Test the API prediction endpoint"""
    print("\n🔮 Testing API prediction endpoint...")
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{BASE_URL}/api/predict", 
                               json=TEST_DATA, 
                               headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API prediction successful:")
            print(f"   Prediction: {data['prediction']} ({'Default' if data['prediction'] == 1 else 'No Default'})")
            print(f"   Probability: {data['probability']:.4f} ({data['probability']*100:.2f}%)")
            print(f"   Risk Level: {data['risk_level']}")
            print(f"   Success: {data['success']}")
            return True
        else:
            print(f"❌ API prediction failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API prediction error: {str(e)}")
        return False

def test_web_interface():
    """Test the web interface (form submission)"""
    print("\n🌐 Testing web interface...")
    try:
        # First, test if the main page loads
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Main page loads successfully")
            
            # Test form submission
            form_data = {
                'person_age': str(TEST_DATA['person_age']),
                'person_income': str(TEST_DATA['person_income']),
                'person_emp_length': str(TEST_DATA['person_emp_length']),
                'person_home_ownership': TEST_DATA['person_home_ownership'],
                'loan_amnt': str(TEST_DATA['loan_amnt']),
                'loan_intent': TEST_DATA['loan_intent'],
                'loan_int_rate': str(TEST_DATA['loan_int_rate']),
                'loan_percent_income': str(TEST_DATA['loan_percent_income']),
                'loan_grade': TEST_DATA['loan_grade'],
                'cb_person_default_on_file': TEST_DATA['cb_person_default_on_file'],
                'cb_person_cred_hist_length': str(TEST_DATA['cb_person_cred_hist_length'])
            }
            
            response = requests.post(f"{BASE_URL}/predict", data=form_data)
            if response.status_code == 200:
                print("✅ Form submission successful")
                print("   Results page loaded successfully")
                return True
            else:
                print(f"❌ Form submission failed with status: {response.status_code}")
                return False
        else:
            print(f"❌ Main page failed to load with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Web interface test error: {str(e)}")
        return False

def test_multiple_predictions():
    """Test multiple predictions with different risk levels"""
    print("\n📊 Testing multiple predictions...")
    
    test_cases = [
        {
            "name": "Low Risk Case",
            "data": {
                "person_age": 35,
                "person_income": 80000,
                "person_emp_length": 10,
                "person_home_ownership": "OWN",
                "loan_amnt": 5000,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_int_rate": 8.5,
                "loan_percent_income": 0.0625,
                "loan_grade": "A",
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 15
            }
        },
        {
            "name": "High Risk Case",
            "data": {
                "person_age": 22,
                "person_income": 25000,
                "person_emp_length": 0.5,
                "person_home_ownership": "RENT",
                "loan_amnt": 20000,
                "loan_intent": "VENTURE",
                "loan_int_rate": 18.0,
                "loan_percent_income": 0.8,
                "loan_grade": "F",
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 2
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n   Testing {case['name']}...")
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(f"{BASE_URL}/api/predict", 
                                   json=case['data'], 
                                   headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {case['name']} - Risk: {data['risk_level']} ({data['probability']*100:.1f}%)")
            else:
                print(f"   ❌ {case['name']} failed")
        except Exception as e:
            print(f"   ❌ {case['name']} error: {str(e)}")

def main():
    """Run all tests"""
    print("🚀 Credit Risk Prediction Web App - Test Suite")
    print("=" * 60)
    print(f"Testing app at: {BASE_URL}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_health_check():
        tests_passed += 1
    
    if test_api_prediction():
        tests_passed += 1
    
    if test_web_interface():
        tests_passed += 1
    
    # Additional tests
    test_multiple_predictions()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} core tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The web app is working correctly.")
        print("\n🌐 You can now access the app at: http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\n💡 Common issues:")
        print("   - Make sure the Flask app is running: python app.py")
        print("   - Ensure the model file exists: ../best_model_pipeline.pkl")
        print("   - Check if all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 