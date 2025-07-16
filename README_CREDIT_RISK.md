# Credit Risk Identifier 🏦

A comprehensive machine learning web application for credit risk assessment built with Flask and advanced ML models.

![Credit Risk Assessment](https://img.shields.io/badge/ML-Credit%20Risk-blue) ![Flask](https://img.shields.io/badge/Flask-Web%20App-green) ![Python](https://img.shields.io/badge/Python-3.8+-yellow)

## 🚀 Overview

This project provides an end-to-end solution for credit risk prediction using machine learning. It features a user-friendly web interface built with Flask that allows financial institutions to assess loan default risk in real-time.

### Key Features

- **Champion ML Model**: Gradient Boosting Classifier achieving **91.90% accuracy** and **91.85% AUC**
- **Advanced Feature Engineering**: Binning + Encoding approach for optimal performance
- **5-Tier Risk Assessment**: Low, Medium-Low, Medium, Medium-High, High Risk
- **Interactive Web Interface**: Bootstrap-powered responsive design
- **Business Recommendations**: Actionable insights for each risk level
- **Real-time Predictions**: API endpoints for integration
- **Comprehensive Testing**: Full test suite included

## 📊 Model Performance

- **Accuracy**: 91.90%
- **AUC Score**: 91.85%
- **Cross-Validation Score**: 91.67%
- **Feature Engineering**: Binning + Categorical Encoding
- **Model Type**: Gradient Boosting Classifier (No PCA)

## 🏗️ Project Structure

```
Credit-Risk-Identifier/
│
├── Web_App/                          # Flask Web Application
│   ├── app.py                        # Main Flask application
│   ├── requirements.txt              # Python dependencies
│   ├── train_champion_model.py       # Model training script
│   ├── test_app.py                   # Test suite
│   ├── README.md                     # Web app documentation
│   └── templates/
│       ├── index.html               # Input form
│       └── result.html              # Results display
│
├── credit-risk-dataset/              # Dataset
│   └── credit_risk_dataset.csv      # Main dataset (32,581 samples)
│
├── best_model_pipeline.pkl           # Trained model (legacy)
├── modeling_week15_comprehensive.ipynb  # ML development notebook
└── INTEGRATION_SUMMARY.md           # Technical documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sys0507/Credit-Risk-Identifier.git
   cd Credit-Risk-Identifier
   ```

2. **Navigate to Web App**
   ```bash
   cd Web_App
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if needed)
   ```bash
   python train_champion_model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the app**
   Open your browser and go to `http://localhost:5000`

## 📋 Input Features

The model requires 11 key features for prediction:

| Feature | Description | Type |
|---------|-------------|------|
| `person_age` | Age of the person | Numerical |
| `person_income` | Annual income | Numerical |
| `person_emp_length` | Employment length (years) | Numerical |
| `person_home_ownership` | Home ownership status | Categorical |
| `loan_amnt` | Loan amount requested | Numerical |
| `loan_intent` | Purpose of the loan | Categorical |
| `loan_int_rate` | Interest rate | Numerical |
| `loan_percent_income` | Loan as % of income | Numerical |
| `loan_grade` | Loan grade | Categorical |
| `cb_person_default_on_file` | Historical default | Categorical |
| `cb_person_cred_hist_length` | Credit history length | Numerical |

## 🎯 Risk Assessment Levels

| Risk Level | Recommendation | Action |
|------------|----------------|---------|
| **Low Risk** | ✅ APPROVE | Standard processing |
| **Medium-Low Risk** | ✅ APPROVE | Consider terms adjustment |
| **Medium Risk** | ⚠️ REVIEW | Additional verification |
| **Medium-High Risk** | ⚠️ CAUTION | Enhanced due diligence |
| **High Risk** | ❌ HIGH RISK | Reject or require collateral |

## 🔧 API Endpoints

- `GET /` - Web interface
- `POST /predict` - Form submission
- `POST /api/predict` - JSON API
- `GET /health` - Health check

### API Usage Example

```python
import requests

data = {
    "person_age": 25,
    "person_income": 50000,
    "person_emp_length": 3,
    "person_home_ownership": "RENT",
    "loan_amnt": 10000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.2,
    "loan_grade": "B",
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5
}

response = requests.post("http://localhost:5000/api/predict", json=data)
result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.2%}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_app.py
```

Tests cover:
- Model loading and prediction accuracy
- Form validation and error handling
- API endpoint functionality
- Risk level classification
- Template rendering

## 📈 Model Development

The champion model was selected from 54 different model variations tested across:
- **3 Feature Engineering Approaches**: Raw Features, Log Transform, Binning+Encoding
- **2 PCA Options**: With/Without PCA
- **9 ML Algorithms**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LGBM, CatBoost, Gradient Boosting, SVM, KNN

See `modeling_week15_comprehensive.ipynb` for complete analysis.

## 🛠️ Technical Stack

- **Backend**: Flask (Python)
- **ML Framework**: scikit-learn, XGBoost, CatBoost
- **Frontend**: Bootstrap 5, Chart.js
- **Data Processing**: pandas, numpy
- **Model Serving**: pickle, joblib

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

- **Author**: sys0507
- **Repository**: [Credit-Risk-Identifier](https://github.com/sys0507/Credit-Risk-Identifier)
- **Issues**: [Report bugs or request features](https://github.com/sys0507/Credit-Risk-Identifier/issues)

## 🏆 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 91.90% |
| Precision | 91.23% |
| Recall | 92.15% |
| F1-Score | 91.69% |
| AUC-ROC | 91.85% |

---

**Built with ❤️ for better financial decision making** 