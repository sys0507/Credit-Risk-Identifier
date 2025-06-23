import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import logging
from datetime import datetime
import json
import re
from fake_useragent import UserAgent
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    filename='web_crawler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LoanDataCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self.setup_selenium()
        
        # Updated data sources for different loan types
        self.loan_sources = {
            'EDUCATION': [
                'https://www.nerdwallet.com/best/loans/student-loans/private-student-loans',
                'https://www.bankrate.com/loans/student-loans/rates/',
                'https://www.credible.com/blog/statistics/student-loan-debt-statistics/'
            ],
            'MEDICAL': [
                'https://www.bankrate.com/loans/personal-loans/medical-loans/',
                'https://www.nerdwallet.com/best/loans/personal-loans/medical-loans',
                'https://www.lendingtree.com/personal/medical-loans/'
            ],
            'VENTURE': [
                'https://www.bankrate.com/loans/business-loans/rates/',
                'https://www.nerdwallet.com/best/small-business/small-business-loans',
                'https://www.fundera.com/business-loans/guides/business-loan-statistics'
            ],
            'HOMEIMPROVEMENT': [
                'https://www.bankrate.com/loans/home-improvement/rates/',
                'https://www.nerdwallet.com/best/loans/personal-loans/home-improvement-loans',
                'https://www.lendingtree.com/home/renovation/home-improvement-loan-rates/'
            ],
            'DEBTCONSOLIDATION': [
                'https://www.bankrate.com/loans/personal-loans/debt-consolidation-loans/',
                'https://www.nerdwallet.com/best/loans/personal-loans/debt-consolidation-loans',
                'https://www.lendingtree.com/personal/debt-consolidation-loans/'
            ],
            'PERSONAL': [
                'https://www.bankrate.com/loans/personal-loans/rates/',
                'https://www.nerdwallet.com/best/loans/personal-loans',
                'https://www.lendingtree.com/personal-loans/'
            ]
        }
        
        # Updated data sources for home ownership types
        self.ownership_sources = {
            'RENT': [
                'https://www.bankrate.com/real-estate/rent-vs-buy-calculator/',
                'https://www.nerdwallet.com/mortgages/rent-vs-buy-calculator',
                'https://www.lendingtree.com/home/mortgage/rent-vs-buy/'
            ],
            'MORTGAGE': [
                'https://www.bankrate.com/mortgages/mortgage-rates/',
                'https://www.nerdwallet.com/mortgages/mortgage-rates',
                'https://www.lendingtree.com/home/mortgage/rates/'
            ],
            'OWN': [
                'https://www.bankrate.com/real-estate/housing-market/',
                'https://www.nerdwallet.com/mortgages/how-much-house-can-i-afford',
                'https://www.lendingtree.com/home/mortgage/'
            ]
        }

    def setup_selenium(self):
        """Setup Selenium with Chrome options"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'user-agent={self.ua.random}')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

    def get_page_content(self, url, use_selenium=False):
        """Get page content using either requests or Selenium"""
        try:
            if use_selenium:
                self.driver.get(url)
                time.sleep(random.uniform(3, 5))  # Longer delay for Selenium
                
                # Scroll down to load dynamic content
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                return self.driver.page_source
            else:
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_loan_stats(self, html_content, loan_type):
        """Extract loan statistics from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        stats = {
            'default_rates': [],
            'interest_rates': [],
            'avg_loan_amounts': [],
            'risk_factors': []
        }
        
        # Extract numbers and percentages
        text = soup.get_text()
        
        # Improved patterns for finding default rates
        default_patterns = [
            r'(\d+\.?\d*)%?\s*(?:default rate|delinquency rate|failure rate)',
            r'(\d+\.?\d*)%?\s*of loans? (?:default|fail)',
            r'default rate of (\d+\.?\d*)%?'
        ]
        
        for pattern in default_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats['default_rates'].extend([float(rate) for rate in matches])
        
        # Improved patterns for finding interest rates
        interest_patterns = [
            r'(\d+\.?\d*)%?\s*(?:APR|interest rate|interest)',
            r'rates? from (\d+\.?\d*)%',
            r'rates? as low as (\d+\.?\d*)%',
            r'average rate of (\d+\.?\d*)%'
        ]
        
        for pattern in interest_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats['interest_rates'].extend([float(rate) for rate in matches])
        
        # Improved patterns for finding loan amounts
        amount_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:thousand|million|billion|k|M|B)?',
            r'average loan (?:amount|size) of \$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'loans? up to \$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats['avg_loan_amounts'].extend([self.parse_amount(amt) for amt in matches])
        
        return stats

    def parse_amount(self, amount_str):
        """Parse string amounts to numbers"""
        try:
            # Remove commas and convert to float
            amount = float(amount_str.replace(',', ''))
            
            # Look for multiplier words
            text = amount_str.lower()
            if 'billion' in text or 'b' in text:
                amount *= 1_000_000_000
            elif 'million' in text or 'm' in text:
                amount *= 1_000_000
            elif 'thousand' in text or 'k' in text:
                amount *= 1_000
                
            return amount
        except ValueError:
            return None

    def crawl_loan_intent_data(self):
        """Crawl data for different loan types"""
        loan_data = {}
        
        for loan_type, urls in self.loan_sources.items():
            loan_stats = {
                'default_rates': [],
                'interest_rates': [],
                'avg_loan_amounts': [],
                'risk_factors': []
            }
            
            for url in urls:
                logging.info(f"Crawling {url} for {loan_type}")
                content = self.get_page_content(url, use_selenium=True)  # Always use Selenium for better reliability
                
                if content:
                    stats = self.extract_loan_stats(content, loan_type)
                    for key in stats:
                        loan_stats[key].extend(stats[key])
                
                # Add delay between requests
                time.sleep(random.uniform(2, 4))
            
            # Calculate aggregated statistics with fallback values
            loan_data[loan_type] = {
                'default_rate': np.median(loan_stats['default_rates']) if loan_stats['default_rates'] else self.get_default_rate(loan_type),
                'avg_interest_rate': np.median(loan_stats['interest_rates']) if loan_stats['interest_rates'] else self.get_default_interest_rate(loan_type),
                'avg_loan_amount': np.median(loan_stats['avg_loan_amounts']) if loan_stats['avg_loan_amounts'] else self.get_default_loan_amount(loan_type)
            }
        
        return loan_data

    def get_default_rate(self, loan_type):
        """Get default default rate for loan type"""
        default_rates = {
            'EDUCATION': 10.1,  # Based on federal student loan default rates
            'MEDICAL': 15.2,    # Higher due to emergency nature
            'VENTURE': 20.0,    # Higher risk for business ventures
            'HOMEIMPROVEMENT': 8.5,  # Lower due to property collateral
            'DEBTCONSOLIDATION': 12.0,  # Medium risk
            'PERSONAL': 10.5    # Average risk
        }
        return default_rates.get(loan_type, 10.0)

    def get_default_interest_rate(self, loan_type):
        """Get default interest rate for loan type"""
        interest_rates = {
            'EDUCATION': 7.5,   # Based on average private student loan rates
            'MEDICAL': 12.0,    # Higher due to urgency
            'VENTURE': 15.0,    # Higher risk premium
            'HOMEIMPROVEMENT': 9.0,  # Moderate due to property backing
            'DEBTCONSOLIDATION': 11.0,  # Based on average personal loan rates
            'PERSONAL': 10.5    # Standard personal loan rate
        }
        return interest_rates.get(loan_type, 10.0)

    def get_default_loan_amount(self, loan_type):
        """Get default loan amount for loan type"""
        loan_amounts = {
            'EDUCATION': 35000,    # Average annual private student loan
            'MEDICAL': 15000,      # Typical medical loan
            'VENTURE': 50000,      # Small business startup
            'HOMEIMPROVEMENT': 25000,  # Common renovation loan
            'DEBTCONSOLIDATION': 20000,  # Average debt consolidation
            'PERSONAL': 15000      # Typical personal loan
        }
        return loan_amounts.get(loan_type, 20000)

    def crawl_home_ownership_data(self):
        """Crawl data for different home ownership types"""
        ownership_data = {}
        
        for ownership_type, urls in self.ownership_sources.items():
            ownership_stats = {
                'stability_scores': [],
                'default_rates': [],
                'market_trends': []
            }
            
            for url in urls:
                logging.info(f"Crawling {url} for {ownership_type}")
                content = self.get_page_content(url, use_selenium=True)
                
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text()
                    
                    # Extract stability indicators with improved patterns
                    stability_patterns = [
                        r'(\d+\.?\d*)%?\s*(?:stable|stability|retention)',
                        r'(\d+\.?\d*)%?\s*(?:stay|remain|keep)',
                        r'stability rate of (\d+\.?\d*)%'
                    ]
                    
                    for pattern in stability_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        ownership_stats['stability_scores'].extend([float(score) for score in matches])
                    
                    # Extract default rates with improved patterns
                    default_patterns = [
                        r'(\d+\.?\d*)%?\s*(?:default|delinquency)',
                        r'(\d+\.?\d*)%?\s*(?:foreclosure|distressed)',
                        r'default rate of (\d+\.?\d*)%'
                    ]
                    
                    for pattern in default_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        ownership_stats['default_rates'].extend([float(rate) for rate in matches])
                
                time.sleep(random.uniform(2, 4))
            
            # Calculate aggregated statistics with fallback values
            ownership_data[ownership_type] = {
                'stability_score': np.median(ownership_stats['stability_scores']) if ownership_stats['stability_scores'] else self.get_default_stability_score(ownership_type),
                'default_rate': np.median(ownership_stats['default_rates']) if ownership_stats['default_rates'] else self.get_default_ownership_default_rate(ownership_type)
            }
        
        return ownership_data

    def get_default_stability_score(self, ownership_type):
        """Get default stability score for ownership type"""
        stability_scores = {
            'OWN': 85.0,      # Highest stability
            'MORTGAGE': 75.0,  # Good stability with some risk
            'RENT': 60.0      # Lower stability
        }
        return stability_scores.get(ownership_type, 70.0)

    def get_default_ownership_default_rate(self, ownership_type):
        """Get default default rate for ownership type"""
        default_rates = {
            'OWN': 2.0,       # Lowest default risk
            'MORTGAGE': 3.5,   # Moderate default risk
            'RENT': 5.0       # Higher default risk
        }
        return default_rates.get(ownership_type, 3.5)

    def save_raw_data(self, loan_data, ownership_data):
        """Save raw crawled data"""
        raw_data = {
            'loan_data': loan_data,
            'ownership_data': ownership_data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('crawled_data_raw.json', 'w') as f:
            json.dump(raw_data, f, indent=2)

    def process_data(self, loan_data, ownership_data):
        """Process raw data into risk scores"""
        processed_data = {
            'loan_intent_risk': {},
            'home_ownership_risk': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Process loan intent data
        for loan_type, stats in loan_data.items():
            risk_score = 0
            if stats['default_rate']:
                risk_score += min(stats['default_rate'] / 25, 1) * 0.4  # Normalize to max 25% default rate
            if stats['avg_interest_rate']:
                risk_score += min(stats['avg_interest_rate'] / 25, 1) * 0.3  # Normalize to max 25% interest rate
            if stats['avg_loan_amount']:
                risk_score += min(stats['avg_loan_amount'] / 100000, 1) * 0.3  # Normalize to $100k
                
            processed_data['loan_intent_risk'][loan_type] = min(risk_score, 1)
        
        # Process home ownership data
        for ownership_type, stats in ownership_data.items():
            risk_score = 0
            if stats['stability_score']:
                risk_score += (100 - stats['stability_score']) / 100 * 0.6  # Higher stability = lower risk
            if stats['default_rate']:
                risk_score += min(stats['default_rate'] / 10, 1) * 0.4  # Normalize to max 10% default rate
                
            processed_data['home_ownership_risk'][ownership_type] = min(risk_score, 1)
        
        # Save processed data
        with open('crawled_data_processed.json', 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        return processed_data

    def enhance_credit_risk_dataset(self, input_file, output_file):
        """Enhance the credit risk dataset with web-crawled features"""
        try:
            # Crawl data
            loan_data = self.crawl_loan_intent_data()
            ownership_data = self.crawl_home_ownership_data()
            
            # Save raw data
            self.save_raw_data(loan_data, ownership_data)
            
            # Process data
            processed_data = self.process_data(loan_data, ownership_data)
            
            # Read original dataset
            df = pd.read_csv(input_file)
            
            # Add new features
            df['loan_intent_risk_score'] = df['loan_intent'].map(processed_data['loan_intent_risk'])
            df['home_ownership_risk_score'] = df['person_home_ownership'].map(processed_data['home_ownership_risk'])
            
            # Calculate combined risk score
            df['combined_risk_score'] = (
                df['loan_intent_risk_score'] * 0.6 +
                df['home_ownership_risk_score'] * 0.4
            )
            
            # Save enhanced dataset
            df.to_csv(output_file, index=False)
            logging.info(f"Enhanced dataset saved to {output_file}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error enhancing dataset: {str(e)}")
            return None
        finally:
            self.driver.quit()

def main():
    """Main function to enhance the dataset with web-crawled features"""
    try:
        crawler = LoanDataCrawler()
        
        # Define input and output files
        input_file = 'credit-risk-dataset/credit_risk_dataset.csv'
        output_file = 'credit-risk-dataset/enhanced_credit_risk_dataset.csv'
        
        # Enhance the dataset
        enhanced_df = crawler.enhance_credit_risk_dataset(input_file, output_file)
        
        if enhanced_df is not None:
            logging.info("Dataset enhancement completed successfully")
            
            # Print some statistics about the new features
            print("\nNew Feature Statistics:")
            print("\nLoan Intent Risk Score Statistics:")
            print(enhanced_df['loan_intent_risk_score'].describe())
            print("\nHome Ownership Risk Score Statistics:")
            print(enhanced_df['home_ownership_risk_score'].describe())
            print("\nCombined Risk Score Statistics:")
            print(enhanced_df['combined_risk_score'].describe())
            
            return enhanced_df
        else:
            logging.error("Failed to enhance dataset")
            return None
            
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        return None

if __name__ == "__main__":
    main() 