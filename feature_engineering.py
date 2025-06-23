import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime

logging.basicConfig(
    filename='feature_engineering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CreditRiskFeatureEngineer:
    def __init__(self):
        # Risk weights based on historical credit data and industry standards
        self.loan_intent_weights = {
            'EDUCATION': {
                'base_risk': 0.4,  # Lower risk due to potential ROI
                'default_history_weight': 0.3,
                'income_ratio_weight': 0.3
            },
            'MEDICAL': {
                'base_risk': 0.5,  # Medium risk due to necessity
                'default_history_weight': 0.25,
                'income_ratio_weight': 0.25
            },
            'VENTURE': {
                'base_risk': 0.7,  # Higher risk due to business uncertainty
                'default_history_weight': 0.15,
                'income_ratio_weight': 0.15
            },
            'PERSONAL': {
                'base_risk': 0.45,  # Medium-low risk
                'default_history_weight': 0.3,
                'income_ratio_weight': 0.25
            },
            'DEBTCONSOLIDATION': {
                'base_risk': 0.6,  # Medium-high risk
                'default_history_weight': 0.2,
                'income_ratio_weight': 0.2
            },
            'HOMEIMPROVEMENT': {
                'base_risk': 0.35,  # Lower risk due to asset improvement
                'default_history_weight': 0.35,
                'income_ratio_weight': 0.3
            }
        }
        
        self.home_ownership_weights = {
            'RENT': {
                'base_stability': 0.4,
                'income_weight': 0.3,
                'employment_weight': 0.3
            },
            'MORTGAGE': {
                'base_stability': 0.6,
                'income_weight': 0.2,
                'employment_weight': 0.2
            },
            'OWN': {
                'base_stability': 0.8,
                'income_weight': 0.1,
                'employment_weight': 0.1
            },
            'OTHER': {
                'base_stability': 0.3,
                'income_weight': 0.35,
                'employment_weight': 0.35
            }
        }

    def calculate_loan_intent_risk(self, row):
        """Calculate risk score based on loan intent and related factors"""
        intent = row['loan_intent']
        weights = self.loan_intent_weights.get(intent, self.loan_intent_weights['PERSONAL'])
        
        # Calculate component scores
        default_history_score = 1.0 if row['cb_person_default_on_file'] == 'Y' else 0.0
        income_ratio_score = min(row['loan_percent_income'], 1.0)
        
        # Combine scores using weights
        risk_score = (
            weights['base_risk'] +
            (default_history_score * weights['default_history_weight']) +
            (income_ratio_score * weights['income_ratio_weight'])
        )
        
        return min(risk_score, 1.0)

    def calculate_home_ownership_risk(self, row):
        """Calculate risk score based on home ownership and related factors"""
        ownership = row['person_home_ownership']
        weights = self.home_ownership_weights.get(ownership, self.home_ownership_weights['OTHER'])
        
        # Normalize income (assuming max income of 200000)
        income_score = min(row['person_income'] / 200000, 1.0)
        
        # Normalize employment length (assuming max length of 30 years)
        emp_length_score = min(row['person_emp_length'] / 30, 1.0) if pd.notnull(row['person_emp_length']) else 0.5
        
        # Calculate stability score
        stability_score = (
            weights['base_stability'] +
            (income_score * weights['income_weight']) +
            (emp_length_score * weights['employment_weight'])
        )
        
        return min(1.0 - stability_score, 1.0)  # Convert stability to risk (higher stability = lower risk)

    def calculate_credit_history_score(self, row):
        """Calculate credit history score based on available credit information"""
        # Base score from credit history length (assuming max length of 30 years)
        base_score = min(row['cb_person_cred_hist_length'] / 30, 1.0)
        
        # Adjust for default history
        if row['cb_person_default_on_file'] == 'Y':
            base_score *= 0.5
        
        return base_score

    def calculate_loan_grade_risk(self, row):
        """Calculate risk score based on loan grade"""
        grade_weights = {
            'A': 0.1,
            'B': 0.3,
            'C': 0.5,
            'D': 0.7,
            'E': 0.8,
            'F': 0.9,
            'G': 1.0
        }
        return grade_weights.get(row['loan_grade'], 0.5)

    def enhance_dataset(self, input_file, output_file):
        """
        Enhance the credit risk dataset with engineered features
        
        Args:
            input_file (str): Path to the input CSV file
            output_file (str): Path to save the enhanced CSV file
        """
        try:
            # Read the original dataset
            df = pd.read_csv(input_file)
            logging.info(f"Read original dataset from {input_file}")
            
            # Calculate basic risk scores
            df['loan_intent_risk_score'] = df.apply(self.calculate_loan_intent_risk, axis=1)
            df['home_ownership_risk_score'] = df.apply(self.calculate_home_ownership_risk, axis=1)
            df['credit_history_score'] = df.apply(self.calculate_credit_history_score, axis=1)
            df['loan_grade_risk_score'] = df.apply(self.calculate_loan_grade_risk, axis=1)
            
            # Calculate age-based risk factor (assuming age range 18-80)
            df['age_risk_factor'] = 1 - ((df['person_age'] - 18) / (80 - 18)).clip(0, 1)
            
            # Calculate income stability score
            df['income_stability_score'] = (df['person_income'] / df['loan_amnt']).clip(0, 1)
            
            # Calculate combined risk score using weighted average
            df['combined_risk_score'] = (
                df['loan_intent_risk_score'] * 0.25 +
                df['home_ownership_risk_score'] * 0.20 +
                df['credit_history_score'] * 0.20 +
                df['loan_grade_risk_score'] * 0.15 +
                df['age_risk_factor'] * 0.10 +
                df['income_stability_score'] * 0.10
            )
            
            # Normalize the combined risk score to 0-1 range
            scaler = MinMaxScaler()
            df['combined_risk_score'] = scaler.fit_transform(df[['combined_risk_score']])
            
            # Add interaction features
            df['intent_ownership_interaction'] = df['loan_intent_risk_score'] * df['home_ownership_risk_score']
            df['credit_grade_interaction'] = df['credit_history_score'] * df['loan_grade_risk_score']
            
            # Save enhanced dataset
            df.to_csv(output_file, index=False)
            logging.info(f"Enhanced dataset saved to {output_file}")
            
            # Print feature statistics
            print("\nNew Feature Statistics:")
            new_features = [
                'loan_intent_risk_score', 
                'home_ownership_risk_score',
                'credit_history_score',
                'loan_grade_risk_score',
                'age_risk_factor',
                'income_stability_score',
                'combined_risk_score',
                'intent_ownership_interaction',
                'credit_grade_interaction'
            ]
            
            for feature in new_features:
                print(f"\n{feature} Statistics:")
                print(df[feature].describe())
            
            return df
            
        except Exception as e:
            logging.error(f"Error enhancing dataset: {str(e)}")
            return None

def main():
    """Main function to enhance the dataset with new features"""
    try:
        engineer = CreditRiskFeatureEngineer()
        
        # Define input and output files
        input_file = 'credit-risk-dataset/credit_risk_dataset.csv'
        output_file = 'credit-risk-dataset/enhanced_credit_risk_dataset.csv'
        
        # Enhance the dataset
        enhanced_df = engineer.enhance_dataset(input_file, output_file)
        
        if enhanced_df is not None:
            logging.info("Dataset enhancement completed successfully")
            return enhanced_df
        else:
            logging.error("Failed to enhance dataset")
            return None
            
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        return None

if __name__ == "__main__":
    main() 