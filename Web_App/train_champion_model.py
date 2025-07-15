#!/usr/bin/env python3
"""
Train a fresh champion model that's compatible with the current environment.
Based on the champion model: Gradient Boosting with binning+encoding, no PCA
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration based on the notebook
NUMERICAL_BINNING_CONFIG = {
    'person_age': {
        'bins': [0, 25, 35, 50, 65, 100],
        'labels': ['Young', 'Adult', 'Middle', 'Senior', 'Elder']
    },
    'person_income': {
        'bins': [0, 30000, 50000, 80000, 120000, float('inf')],
        'labels': ['Low', 'Medium', 'High', 'Very_High', 'Ultra_High']
    },
    'person_emp_length': {
        'bins': [0, 1, 5, 10, 20, float('inf')],
        'labels': ['New', 'Short', 'Medium', 'Long', 'Very_Long']
    },
    'loan_amnt': {
        'bins': [0, 5000, 10000, 15000, 25000, float('inf')],
        'labels': ['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large']
    },
    'loan_int_rate': {
        'bins': [0, 8, 12, 16, 20, float('inf')],
        'labels': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
    },
    'loan_percent_income': {
        'bins': [0, 0.1, 0.2, 0.3, 0.5, float('inf')],
        'labels': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
    },
    'cb_person_cred_hist_length': {
        'bins': [0, 5, 10, 20, float('inf')],
        'labels': ['Short', 'Medium', 'Long', 'Very_Long']
    }
}

def load_and_prepare_data():
    """Load and prepare the credit risk dataset"""
    
    # Try to find the dataset
    possible_paths = [
        '../credit-risk-dataset/credit_risk_dataset.csv',
        '../processed_credit_risk_dataset.csv',
        '../enhanced_credit_risk_dataset.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError("Could not find credit risk dataset. Please ensure it exists in the Group_project directory.")
    
    print(f"📊 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def apply_binning_encoding(df):
    """Apply binning and encoding to the dataset"""
    
    print("🔧 Applying binning and encoding...")
    
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Apply numerical binning
    for col, config in NUMERICAL_BINNING_CONFIG.items():
        if col in df_processed.columns:
            df_processed[col] = pd.cut(
                df_processed[col], 
                bins=config['bins'], 
                labels=config['labels'], 
                include_lowest=True
            )
            print(f"  ✅ Binned {col}: {df_processed[col].value_counts().to_dict()}")
        else:
            print(f"  ⚠️ Column {col} not found in dataset")
    
    # Convert categorical columns to string
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
    
    # Apply one-hot encoding for categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Get all categorical columns (including binned ones)
    cat_columns = []
    for col in df_processed.columns:
        if col != 'loan_status' and (df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'category'):
            cat_columns.append(col)
    
    print(f"  📝 Categorical columns to encode: {cat_columns}")
    
    if cat_columns:
        # Fit and transform categorical columns
        encoded_features = encoder.fit_transform(df_processed[cat_columns])
        
        # Get feature names
        feature_names = encoder.get_feature_names_out(cat_columns)
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df_processed.index)
        
        # Drop original categorical columns and add encoded ones
        df_processed = df_processed.drop(columns=cat_columns)
        df_processed = pd.concat([df_processed, encoded_df], axis=1)
        
        print(f"  ✅ Encoded features shape: {encoded_features.shape}")
        print(f"  ✅ Final processed shape: {df_processed.shape}")
        
        return df_processed, encoder
    else:
        return df_processed, None

def train_champion_model():
    """Train the champion model"""
    
    print("🚀 TRAINING CHAMPION MODEL")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Check if target column exists
    if 'loan_status' not in df.columns:
        raise ValueError("Target column 'loan_status' not found in dataset")
    
    # Apply binning and encoding
    df_processed, encoder = apply_binning_encoding(df)
    
    # Separate features and target
    X = df_processed.drop('loan_status', axis=1)
    y = df_processed['loan_status']
    
    print(f"📊 Features shape: {X.shape}")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # Define Gradient Boosting parameters (reduced grid for faster training)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2']
    }
    
    # Create model
    gb_model = GradientBoostingClassifier(random_state=42)
    
    # Grid search with cross-validation
    print("🔍 Starting GridSearch...")
    grid_search = GridSearchCV(
        gb_model, 
        param_grid, 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print(f"🏆 Best parameters: {grid_search.best_params_}")
    print(f"🏆 Best CV score: {grid_search.best_score_:.4f}")
    
    # Test the model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"📊 Test ROC-AUC: {test_auc:.4f}")
    
    # Create model package
    model_package = {
        'model': best_model,
        'encoder': encoder,
        'feature_names': list(X.columns),
        'binning_config': NUMERICAL_BINNING_CONFIG,
        'performance': {
            'cv_score': grid_search.best_score_,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc
        }
    }
    
    # Save the model
    model_path = '../best_model_pipeline.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Model performance - AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
    
    return model_package

if __name__ == "__main__":
    try:
        model_package = train_champion_model()
        print("🎉 Champion model training completed successfully!")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc() 