import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import random
import logging
from datetime import datetime
import json
import re
from fake_useragent import UserAgent

# Configure logging
logging.basicConfig(
    filename='employment_crawler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EmploymentDataCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self.setup_selenium()
        
        # Data sources for employment statistics
        self.sources = {
            'bls': [
                'https://www.bls.gov/news.release/tenure.nr0.htm',
                'https://www.bls.gov/news.release/jolts.t04.htm'
            ],
            'linkedin': [
                'https://www.linkedin.com/business/talent/blog/talent-strategy/employee-tenure-trends',
                'https://www.linkedin.com/pulse/employee-tenure-industry-2023-data-insights-trends'
            ],
            'glassdoor': [
                'https://www.glassdoor.com/research/employee-tenure-trends/',
                'https://www.glassdoor.com/research/job-market-trends/'
            ]
        }

    def setup_selenium(self):
        """Setup Selenium with Chrome options"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'user-agent={self.ua.random}')
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

    def get_page_content(self, url, use_selenium=False):
        """Get page content using either requests or Selenium"""
        try:
            if use_selenium:
                self.driver.get(url)
                time.sleep(random.uniform(2, 4))
                return self.driver.page_source
            else:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_employment_stats(self, html_content):
        """Extract employment statistics from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        stats = {
            'avg_tenure': [],
            'turnover_rates': [],
            'industry_norms': {}
        }
        
        text = soup.get_text()
        
        # Extract average tenure
        tenure_patterns = [
            r'(\d+\.?\d*)\s*(?:years?|yrs?)(?:\s+average|\s+median)?\s+tenure',
            r'average\s+tenure\s+of\s+(\d+\.?\d*)\s*(?:years?|yrs?)',
            r'median\s+tenure\s+(?:of\s+)?(\d+\.?\d*)\s*(?:years?|yrs?)'
        ]
        
        for pattern in tenure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats['avg_tenure'].extend([float(tenure) for tenure in matches])
        
        # Extract turnover rates
        turnover_patterns = [
            r'(\d+\.?\d*)%?\s*(?:turnover|attrition)\s+rate',
            r'turnover\s+rate\s+of\s+(\d+\.?\d*)%?',
            r'attrition\s+rate\s+of\s+(\d+\.?\d*)%?'
        ]
        
        for pattern in turnover_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats['turnover_rates'].extend([float(rate) for rate in matches])
        
        return stats

    def get_default_stats(self):
        """Get default employment statistics based on general research"""
        return {
            'avg_tenure': 4.1,  # Years, based on BLS data
            'turnover_rate': 15.0,  # Percentage, industry average
            'risk_thresholds': {
                'very_low': 8.0,    # Years
                'low': 5.0,         # Years
                'medium': 3.0,      # Years
                'high': 1.0,        # Years
                'very_high': 0.5    # Years
            }
        }

    def crawl_employment_data(self):
        """Crawl employment data from various sources"""
        employment_data = {
            'avg_tenure': [],
            'turnover_rates': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for source, urls in self.sources.items():
            for url in urls:
                logging.info(f"Crawling {url}")
                content = self.get_page_content(url, use_selenium='linkedin' in url or 'glassdoor' in url)
                
                if content:
                    stats = self.extract_employment_stats(content)
                    employment_data['avg_tenure'].extend(stats['avg_tenure'])
                    employment_data['turnover_rates'].extend(stats['turnover_rates'])
                
                time.sleep(random.uniform(2, 4))
        
        # Calculate aggregated statistics
        default_stats = self.get_default_stats()
        
        processed_data = {
            'avg_tenure': np.median(employment_data['avg_tenure']) if employment_data['avg_tenure'] else default_stats['avg_tenure'],
            'turnover_rate': np.median(employment_data['turnover_rates']) if employment_data['turnover_rates'] else default_stats['turnover_rate'],
            'risk_thresholds': default_stats['risk_thresholds']
        }
        
        return processed_data

    def calculate_employment_risk_score(self, emp_length, stats):
        """Calculate risk score based on employment length"""
        if emp_length >= stats['risk_thresholds']['very_low']:
            return 0.1  # Very low risk
        elif emp_length >= stats['risk_thresholds']['low']:
            return 0.3  # Low risk
        elif emp_length >= stats['risk_thresholds']['medium']:
            return 0.5  # Medium risk
        elif emp_length >= stats['risk_thresholds']['high']:
            return 0.7  # High risk
        else:
            return 0.9  # Very high risk

    def enhance_credit_risk_dataset(self, input_file, output_file):
        """Enhance the credit risk dataset with employment risk features"""
        try:
            # Crawl employment data
            employment_stats = self.crawl_employment_data()
            
            # Save raw employment data
            with open('employment_data_raw.json', 'w') as f:
                json.dump(employment_stats, f, indent=2)
            
            # Read original dataset
            df = pd.read_csv(input_file)
            
            # Add employment risk score
            df['emp_length_risk_score'] = df['person_emp_length'].apply(
                lambda x: self.calculate_employment_risk_score(x/12 if x else 0, employment_stats)
            )
            
            # Add employment length ratio (compared to average)
            df['emp_length_ratio'] = df['person_emp_length'].apply(
                lambda x: (x/12) / employment_stats['avg_tenure'] if x else 0
            )
            
            # Add turnover risk indicator
            df['emp_turnover_risk'] = df['person_emp_length'].apply(
                lambda x: 1 if x and x/12 < 1 else 0
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
    """Main function to enhance the dataset with employment data"""
    try:
        crawler = EmploymentDataCrawler()
        
        input_file = 'credit-risk-dataset/credit_risk_dataset.csv'
        output_file = 'credit-risk-dataset/employment_enhanced_dataset.csv'
        
        enhanced_df = crawler.enhance_credit_risk_dataset(input_file, output_file)
        
        if enhanced_df is not None:
            logging.info("Dataset enhancement completed successfully")
            
            print("\nNew Employment Feature Statistics:")
            print("\nEmployment Length Risk Score Statistics:")
            print(enhanced_df['emp_length_risk_score'].describe())
            print("\nEmployment Length Ratio Statistics:")
            print(enhanced_df['emp_length_ratio'].describe())
            print("\nTurnover Risk Statistics:")
            print(enhanced_df['emp_turnover_risk'].value_counts())
            
            return enhanced_df
        else:
            logging.error("Failed to enhance dataset")
            return None
            
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        return None

if __name__ == "__main__":
    main() 