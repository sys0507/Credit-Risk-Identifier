import json
import pandas as pd

# Read the raw JSON data
with open('crawled_data_raw.json', 'r') as f:
    raw_data = json.load(f)

# Convert loan data to DataFrame
loan_data = pd.DataFrame.from_dict(raw_data['loan_data'], orient='index')
loan_data.index.name = 'loan_type'
loan_data.reset_index(inplace=True)
loan_data.to_csv('loan_data_raw.csv', index=False)

# Convert ownership data to DataFrame
ownership_data = pd.DataFrame.from_dict(raw_data['ownership_data'], orient='index')
ownership_data.index.name = 'ownership_type'
ownership_data.reset_index(inplace=True)
ownership_data.to_csv('ownership_data_raw.csv', index=False)

# Read the processed JSON data
with open('crawled_data_processed.json', 'r') as f:
    processed_data = json.load(f)

# Convert processed loan risk data to DataFrame
loan_risk = pd.DataFrame.from_dict(processed_data['loan_intent_risk'], orient='index')
loan_risk.index.name = 'loan_type'
loan_risk.columns = ['risk_score']
loan_risk.reset_index(inplace=True)
loan_risk.to_csv('loan_risk_processed.csv', index=False)

# Convert processed ownership risk data to DataFrame
ownership_risk = pd.DataFrame.from_dict(processed_data['home_ownership_risk'], orient='index')
ownership_risk.index.name = 'ownership_type'
ownership_risk.columns = ['risk_score']
ownership_risk.reset_index(inplace=True)
ownership_risk.to_csv('ownership_risk_processed.csv', index=False)

# Merge raw and processed data for analysis
loan_analysis = loan_data.merge(loan_risk, on='loan_type')
loan_analysis.to_csv('loan_analysis_combined.csv', index=False)

ownership_analysis = ownership_data.merge(ownership_risk, on='ownership_type')
ownership_analysis.to_csv('ownership_analysis_combined.csv', index=False) 