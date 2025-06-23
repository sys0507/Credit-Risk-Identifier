import json
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def json_to_csv():
    try:
        # Read the raw JSON data
        with open('crawled_data_raw.json', 'r') as f:
            raw_data = json.load(f)
        
        # Convert loan data to DataFrame
        loan_data = []
        for loan_type, stats in raw_data['loan_data'].items():
            loan_data.append({
                'loan_type': loan_type,
                'default_rate': stats['default_rate'],
                'avg_interest_rate': stats['avg_interest_rate'],
                'avg_loan_amount': stats['avg_loan_amount']
            })
        loan_df = pd.DataFrame(loan_data)
        
        # Convert ownership data to DataFrame
        ownership_data = []
        for ownership_type, stats in raw_data['ownership_data'].items():
            ownership_data.append({
                'ownership_type': ownership_type,
                'stability_score': stats['stability_score'],
                'default_rate': stats['default_rate']
            })
        ownership_df = pd.DataFrame(ownership_data)
        
        # Read employment data if available
        try:
            with open('employment_data_raw.json', 'r') as f:
                emp_data = json.load(f)
            
            # Convert employment data to DataFrame
            emp_df = pd.DataFrame([{
                'metric': 'Average Tenure (years)',
                'value': emp_data['avg_tenure']
            }, {
                'metric': 'Turnover Rate (%)',
                'value': emp_data['turnover_rate']
            }])
            
            # Create risk thresholds DataFrame
            risk_df = pd.DataFrame([{
                'risk_level': level,
                'tenure_threshold_years': value
            } for level, value in emp_data['risk_thresholds'].items()])
            
            # Save employment data to CSV
            emp_df.to_csv('employment_stats_raw.csv', index=False)
            risk_df.to_csv('employment_risk_thresholds.csv', index=False)
            logging.info("Employment data exported to CSV files")
            
            print("\nEmployment Statistics Preview:")
            print(emp_df)
            print("\nRisk Thresholds Preview:")
            print(risk_df)
            
        except FileNotFoundError:
            logging.warning("Employment data file not found")
        
        # Save loan and ownership data to CSV
        loan_df.to_csv('loan_data_raw.csv', index=False)
        ownership_df.to_csv('ownership_data_raw.csv', index=False)
        logging.info("Loan and ownership data exported to CSV files")
        
        print("\nLoan Data Preview:")
        print(loan_df)
        print("\nOwnership Data Preview:")
        print(ownership_df)
        
    except Exception as e:
        logging.error(f"Error converting data: {str(e)}")

if __name__ == "__main__":
    json_to_csv() 