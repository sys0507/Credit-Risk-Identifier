import pandas as pd

# Read the dataset
df = pd.read_csv('credit-risk-dataset/credit_risk_dataset.csv')

# Print unique values
print('Loan Intents:', sorted(df['loan_intent'].unique()))
print('\nHome Ownership Types:', sorted(df['person_home_ownership'].unique()))

# Print some statistics
print('\nLoan Intent Distribution:')
print(df['loan_intent'].value_counts())
print('\nHome Ownership Distribution:')
print(df['person_home_ownership'].value_counts()) 