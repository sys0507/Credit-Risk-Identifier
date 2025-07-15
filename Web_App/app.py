from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import required libraries for feature engineering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

# Configuration
MODEL_PATH = '../best_model_pipeline.pkl'

# Global variables to store model and preprocessing objects
model_package = None
is_model_loaded = False

def load_model():
    """Load the trained model package"""
    global model_package, is_model_loaded
    
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model_package = pickle.load(f)
            is_model_loaded = True
            print("✅ Model loaded successfully!")
            print(f"Model type: {type(model_package['model'])}")
            print(f"Performance: AUC={model_package['performance']['test_auc']:.4f}, Accuracy={model_package['performance']['test_accuracy']:.4f}")
            return True
        else:
            print(f"❌ Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocess input data for prediction"""
    if not is_model_loaded:
        raise Exception("Model not loaded")
    
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # Apply binning to numerical features
    binning_config = model_package['binning_config']
    
    for col, config in binning_config.items():
        if col in df.columns:
            df[col] = pd.cut(
                df[col], 
                bins=config['bins'], 
                labels=config['labels'], 
                include_lowest=True
            )
    
    # Convert categorical columns to string
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Apply one-hot encoding using the trained encoder
    encoder = model_package['encoder']
    
    # Get categorical columns (same as training)
    cat_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            cat_columns.append(col)
    
    if cat_columns and encoder is not None:
        # Transform categorical columns
        encoded_features = encoder.transform(df[cat_columns])
        
        # Get feature names
        feature_names = encoder.get_feature_names_out(cat_columns)
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
        
        # Drop original categorical columns and add encoded ones
        df = df.drop(columns=cat_columns)
        df = pd.concat([df, encoded_df], axis=1)
    
    # Ensure all training features are present
    training_features = model_package['feature_names']
    for feature in training_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only training features in the same order
    df = df[training_features]
    
    return df

def predict_loan_risk(data):
    """Make prediction using the loaded model"""
    if not is_model_loaded:
        raise Exception("Model not loaded")
    
    # Preprocess the input
    processed_data = preprocess_input(data)
    
    # Make prediction
    model = model_package['model']
    prediction = model.predict(processed_data)[0]
    prediction_proba = model.predict_proba(processed_data)[0]
    
    # Get probability of default (class 1)
    risk_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
    
    return {
        'prediction': int(prediction),
        'risk_probability': float(risk_probability),
        'risk_percentage': float(risk_probability * 100)
    }

def get_risk_level(risk_probability):
    """Determine risk level based on probability"""
    if risk_probability < 0.2:
        return "Low Risk"
    elif risk_probability < 0.4:
        return "Medium-Low Risk"
    elif risk_probability < 0.6:
        return "Medium Risk"
    elif risk_probability < 0.8:
        return "Medium-High Risk"
    else:
        return "High Risk"

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        "Low Risk": "#28a745",
        "Medium-Low Risk": "#6c757d", 
        "Medium Risk": "#ffc107",
        "Medium-High Risk": "#fd7e14",
        "High Risk": "#dc3545"
    }
    return colors.get(risk_level, "#6c757d")

def get_business_recommendation(risk_level, risk_probability):
    """Get business recommendation based on risk level"""
    if risk_level == "Low Risk":
        return "✅ **APPROVE** - Proceed with standard loan terms and conditions."
    elif risk_level == "Medium-Low Risk":
        return "✅ **APPROVE** - Consider standard to slightly elevated interest rates."
    elif risk_level == "Medium Risk":
        return "⚠️ **REVIEW** - Detailed manual review required. Consider additional collateral or co-signer."
    elif risk_level == "Medium-High Risk":
        return "⚠️ **CAUTION** - Proceed with caution. Require substantial collateral and higher interest rates."
    else:
        return "❌ **HIGH RISK** - Consider rejection or require significant collateral and guarantees."

# Initialize model on startup
load_model()

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and make prediction"""
    if not is_model_loaded:
        return render_template('index.html', error="Model not loaded. Please try again later.")
    
    try:
        # Get form data
        data = {
            'person_age': float(request.form['person_age']),
            'person_income': float(request.form['person_income']),
            'person_home_ownership': request.form['person_home_ownership'],
            'person_emp_length': float(request.form['person_emp_length']),
            'loan_intent': request.form['loan_intent'],
            'loan_grade': request.form['loan_grade'],
            'loan_amnt': float(request.form['loan_amnt']),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'cb_person_default_on_file': request.form['cb_person_default_on_file'],
            'cb_person_cred_hist_length': float(request.form['cb_person_cred_hist_length'])
        }
        
        # Make prediction
        prediction_result = predict_loan_risk(data)
        
        # Get risk level and recommendations
        risk_level = get_risk_level(prediction_result['risk_probability'])
        risk_color = get_risk_color(risk_level)
        business_recommendation = get_business_recommendation(risk_level, prediction_result['risk_probability'])
        
        # Create result object in the format expected by the template
        result = {
            'prediction': prediction_result['prediction'],
            'prediction_text': 'Default' if prediction_result['prediction'] == 1 else 'No Default',
            'probability': prediction_result['risk_probability'],
            'probability_percent': f"{prediction_result['risk_percentage']:.2f}%",
            'risk_level': risk_level,
            'risk_color': 'success' if risk_level == 'Low Risk' else 
                         'info' if risk_level == 'Medium-Low Risk' else
                         'warning' if risk_level == 'Medium Risk' else
                         'danger',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template('result.html', result=result, input_data=data, business_recommendation=business_recommendation)
        
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        return render_template('index.html', error=error_msg)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not is_model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        result = predict_loan_risk(data)
        
        risk_level = get_risk_level(result['risk_probability'])
        business_recommendation = get_business_recommendation(risk_level, result['risk_probability'])
        
        return jsonify({
            'prediction': result['prediction'],
            'risk_probability': result['risk_probability'],
            'risk_percentage': result['risk_percentage'],
            'risk_level': risk_level,
            'business_recommendation': business_recommendation,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': is_model_loaded,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 