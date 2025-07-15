# 🏆 Credit Risk Prediction Web App

A Flask-based web application that uses the champion machine learning model (Gradient Boosting with binning+encoding, no PCA) to predict loan default risk.

## 📋 Features

- **User-friendly form interface** for entering loan application data
- **Real-time prediction** using the champion model
- **Risk visualization** with interactive charts
- **Detailed results** with risk interpretation and recommendations
- **Responsive design** that works on desktop and mobile
- **API endpoint** for programmatic access

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- The trained champion model file: `best_model_pipeline.pkl`

### Installation

1. **Navigate to the Web_App directory:**
   ```bash
   cd Group_project/Web_App
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model file is in the correct location:**
   - The app expects `best_model_pipeline.pkl` in the parent directory (`../best_model_pipeline.pkl`)
   - If your model file is located elsewhere, update the `MODEL_PATH` variable in `app.py`

### Running the Application

1. **Start the Flask app:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Fill out the form with loan application details and get predictions!**

## 🔧 Model Information

### Champion Model Details
- **Algorithm:** Gradient Boosting Classifier
- **Feature Engineering:** Binning + Encoding approach
- **PCA:** Not applied
- **Performance:** ~95% accuracy on test data

### Input Features Required
The model expects the following 11 input features:

#### Personal Information
- **person_age**: Age of the applicant (18-100 years)
- **person_income**: Annual income in dollars
- **person_emp_length**: Employment length in years
- **person_home_ownership**: Home ownership status (RENT, OWN, MORTGAGE, OTHER)

#### Loan Information
- **loan_amnt**: Loan amount in dollars
- **loan_intent**: Purpose of the loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
- **loan_int_rate**: Interest rate as percentage
- **loan_percent_income**: Loan amount as percentage of income (0.0-1.0)
- **loan_grade**: Loan grade (A, B, C, D, E, F, G)

#### Credit History
- **cb_person_default_on_file**: Previous default history (Y/N)
- **cb_person_cred_hist_length**: Credit history length in years

## 🌐 API Usage

The app provides a REST API endpoint for programmatic access:

### Endpoint
```
POST /api/predict
Content-Type: application/json
```

### Request Body Example
```json
{
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
```

### Response Example
```json
{
    "prediction": 0,
    "probability": 0.15,
    "risk_level": "Low Risk",
    "success": true
}
```

## 📊 Risk Interpretation

The model provides probability scores that are interpreted as follows:

- **Low Risk (0-20%)**: Low default probability - approve with standard terms
- **Medium-Low Risk (20-40%)**: Some concerns - consider standard to slightly elevated terms
- **Medium Risk (40-60%)**: Mixed characteristics - detailed review required
- **Medium-High Risk (60-80%)**: Several concerns - proceed with caution
- **High Risk (80-100%)**: High default probability - consider rejection or substantial collateral

## 🔍 Health Check

Check if the app and model are working correctly:

### Endpoint
```
GET /health
```

### Response Example
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-15T10:30:00"
}
```

## 📁 File Structure

```
Web_App/
├── app.py                 # Main Flask application
├── templates/
│   ├── index.html        # Main form page
│   └── result.html       # Results display page
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚨 Important Notes

1. **Model File**: Ensure the trained model file (`best_model_pipeline.pkl`) is available at the specified path
2. **Feature Engineering**: The app automatically applies the same binning and encoding transformations used during training
3. **Security**: This is a demo application. For production use, add proper authentication, input validation, and security measures
4. **Performance**: The model uses ordinal encoding instead of target encoding for single predictions (since target values aren't available)

## 🛠️ Troubleshooting

### Common Issues

1. **Model not loading**: Check if `best_model_pipeline.pkl` exists in the correct location
2. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Port already in use**: Change the port in `app.py` or stop the conflicting process
4. **Prediction errors**: Verify all input fields are properly filled and within valid ranges

## 🤝 Contributing

This web app is designed to work with the champion model from the comprehensive ML pipeline. To modify or extend:

1. Update the feature engineering logic in `preprocess_input()` if needed
2. Modify the HTML templates for UI changes
3. Add new routes in `app.py` for additional functionality

## 📈 Future Enhancements

- Add batch prediction capability
- Implement user authentication
- Add prediction history tracking
- Create admin dashboard for model monitoring
- Add A/B testing for model variants

---

**Powered by Advanced Machine Learning • Champion Model: Gradient Boosting** 