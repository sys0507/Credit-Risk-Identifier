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
from datetime import datetime, timedelta
import json
import re
from fake_useragent import UserAgent
from urllib.parse import urljoin
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from collections import defaultdict

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
        self.setup_requests_session()
        
        # Historical data files
        self.historical_data_file = 'historical_loan_data.json'
        self.historical_stats_file = 'historical_stats.json'
        self.load_historical_data()
        
        # Add data validation thresholds
        self.validation_thresholds = {
            'min_default_rate': 0.1,  # 0.1%
            'max_default_rate': 30.0,  # 30%
            'min_interest_rate': 2.0,  # 2%
            'max_interest_rate': 35.0,  # 35%
            'min_loan_amount': 1000,   # $1,000
            'max_loan_amount': 500000  # $500,000
        }
        
        # Initialize historical statistics
        self.historical_stats = defaultdict(lambda: {
            'default_rates': [],
            'interest_rates': [],
            'loan_amounts': [],
            'timestamps': []
        })
        
        self.load_historical_stats()
        
        # Updated data sources with more reliable sources
        self.loan_sources = {
            'EDUCATION': [
                'https://www.nerdwallet.com/best/loans/student-loans/private-student-loans',
                'https://www.bankrate.com/loans/student-loans/rates/',
                'https://www.credible.com/blog/statistics/student-loan-debt-statistics/',
                'https://studentaid.gov/data-center/student/default',  # Government source
                'https://www.salliemae.com/student-loans/smart-option-student-loan/'  # Direct lender
            ],
            'MEDICAL': [
                'https://www.bankrate.com/loans/personal-loans/medical-loans/',
                'https://www.nerdwallet.com/best/loans/personal-loans/medical-loans',
                'https://www.lendingtree.com/personal/medical-loans/',
                'https://www.patientadvocate.org/explore-our-resources/preventing-medical-debt/',
                'https://www.creditkarma.com/personal-loans/medical-loans'
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

    def setup_requests_session(self):
        """Setup requests session with retry logic"""
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def load_historical_data(self):
        """Load historical loan data for validation"""
        try:
            with open(self.historical_data_file, 'r') as f:
                self.historical_data = json.load(f)
        except FileNotFoundError:
            self.historical_data = {'loan_data': {}, 'ownership_data': {}}

    def save_historical_data(self, new_data: Dict[str, Any]):
        """Save new data to historical records"""
        self.historical_data['loan_data'][datetime.now().isoformat()] = new_data
        with open(self.historical_data_file, 'w') as f:
            json.dump(self.historical_data, f, indent=2)

    def validate_rate(self, rate: float, rate_type: str) -> Optional[float]:
        """Validate extracted rates against thresholds"""
        if rate_type == 'default':
            min_val = self.validation_thresholds['min_default_rate']
            max_val = self.validation_thresholds['max_default_rate']
        else:  # interest
            min_val = self.validation_thresholds['min_interest_rate']
            max_val = self.validation_thresholds['max_interest_rate']
            
        if min_val <= rate <= max_val:
            return rate
        return None

    def validate_loan_amount(self, amount: float) -> Optional[float]:
        """Validate extracted loan amounts against thresholds"""
        min_val = self.validation_thresholds['min_loan_amount']
        max_val = self.validation_thresholds['max_loan_amount']
        
        if min_val <= amount <= max_val:
            return amount
        return None

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

    def get_page_content(self, url: str, use_selenium: bool = False) -> Optional[str]:
        """Get page content with improved error handling and retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if use_selenium:
                    self.driver.get(url)
                    time.sleep(random.uniform(3, 5))
                    
                    # Wait for dynamic content
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    
                    # Scroll for dynamic loading
                    self.driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);"
                    )
                    time.sleep(2)
                    
                    return self.driver.page_source
                else:
                    response = self.session.get(
                        url, 
                        headers=self.headers, 
                        timeout=15
                    )
                    response.raise_for_status()
                    return response.text
                    
            except Exception as e:
                logging.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {url}: {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    # Rotate user agent
                    self.headers['User-Agent'] = self.ua.random
                else:
                    logging.error(f"Failed to fetch {url} after {max_retries} attempts")
        return None

    def extract_loan_stats(self, html_content: str, loan_type: str) -> Dict[str, List[float]]:
        """Extract loan statistics with improved validation"""
        soup = BeautifulSoup(html_content, 'html.parser')
        stats = {
            'default_rates': [],
            'interest_rates': [],
            'avg_loan_amounts': [],
            'risk_factors': []
        }
        
        text = soup.get_text()
        
        # Enhanced patterns with more specific context
        default_patterns = [
            r'(\d+\.?\d*)%?\s*(?:default rate|delinquency rate|failure rate)',
            r'(\d+\.?\d*)%?\s*of loans? (?:default|fail)',
            r'default rate of (\d+\.?\d*)%?',
            r'(?:historical|average|typical) default rate[s]? (?:of|is|are) (\d+\.?\d*)%'
        ]
        
        interest_patterns = [
            r'(\d+\.?\d*)%?\s*(?:APR|interest rate|interest)',
            r'rates? from (\d+\.?\d*)%',
            r'rates? as low as (\d+\.?\d*)%',
            r'average rate of (\d+\.?\d*)%',
            r'(?:fixed|variable) rate[s]? starting at (\d+\.?\d*)%'
        ]
        
        # Extract and validate rates
        for pattern in default_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for rate in matches:
                try:
                    rate_val = float(rate)
                    validated_rate = self.validate_rate(rate_val, 'default')
                    if validated_rate is not None:
                        stats['default_rates'].append(validated_rate)
                except ValueError:
                    continue
        
        for pattern in interest_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for rate in matches:
                try:
                    rate_val = float(rate)
                    validated_rate = self.validate_rate(rate_val, 'interest')
                    if validated_rate is not None:
                        stats['interest_rates'].append(validated_rate)
                except ValueError:
                    continue
        
        # Extract and validate loan amounts
        amount_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:thousand|million|billion|k|M|B)?',
            r'average loan (?:amount|size) of \$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'loans? up to \$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'typical loan size[s]? (?:of|is|are) \$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for amt in matches:
                try:
                    amount = self.parse_amount(amt)
                    if amount:
                        validated_amount = self.validate_loan_amount(amount)
                        if validated_amount is not None:
                            stats['avg_loan_amounts'].append(validated_amount)
                except ValueError:
                    continue
        
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

    def crawl_loan_intent_data(self) -> Dict[str, Dict[str, float]]:
        """Crawl data with parallel processing and validation"""
        loan_data = {}
        
        def process_url(url: str, loan_type: str) -> Dict[str, List[float]]:
            content = self.get_page_content(url, use_selenium=True)
            if content:
                return self.extract_loan_stats(content, loan_type)
            return {
                'default_rates': [],
                'interest_rates': [],
                'avg_loan_amounts': [],
                'risk_factors': []
            }
        
        for loan_type, urls in self.loan_sources.items():
            loan_stats = {
                'default_rates': [],
                'interest_rates': [],
                'avg_loan_amounts': [],
                'risk_factors': []
            }
            
            # Parallel processing of URLs
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_url = {
                    executor.submit(process_url, url, loan_type): url 
                    for url in urls
                }
                
                for future in future_to_url:
                    try:
                        stats = future.result()
                        for key in stats:
                            loan_stats[key].extend(stats[key])
                    except Exception as e:
                        logging.error(f"Error processing URL: {str(e)}")
            
            # Calculate statistics with confidence scores
            loan_data[loan_type] = self.calculate_loan_statistics(
                loan_stats, 
                loan_type
            )
            
            # Add delay between loan types
            time.sleep(random.uniform(1, 2))
        
        # Save to historical data
        self.save_historical_data(loan_data)
        
        return loan_data

    def load_historical_stats(self):
        """Load historical statistics from file"""
        try:
            with open(self.historical_stats_file, 'r') as f:
                stats = json.load(f)
                for loan_type, data in stats.items():
                    self.historical_stats[loan_type] = data
            logging.info("Loaded historical statistics")
        except FileNotFoundError:
            logging.info("No historical statistics found - will create new file")

    def save_historical_stats(self):
        """Save historical statistics to file"""
        with open(self.historical_stats_file, 'w') as f:
            json.dump(dict(self.historical_stats), f, indent=2)
        logging.info("Saved historical statistics")

    def update_historical_stats(self, loan_type: str, stats: Dict[str, Any]):
        """Update historical statistics with new data"""
        timestamp = datetime.now().isoformat()
        
        # Only update if we have high confidence values
        if stats.get('default_rate_confidence', 0) > 0.7:
            self.historical_stats[loan_type]['default_rates'].append(stats['default_rate'])
            self.historical_stats[loan_type]['timestamps'].append(timestamp)
        
        if stats.get('interest_rate_confidence', 0) > 0.7:
            self.historical_stats[loan_type]['interest_rates'].append(stats['avg_interest_rate'])
        
        if stats.get('loan_amount_confidence', 0) > 0.7:
            self.historical_stats[loan_type]['loan_amounts'].append(stats['avg_loan_amount'])
        
        # Keep only last 90 days of data
        self.prune_old_data(loan_type)
        
    def prune_old_data(self, loan_type: str):
        """Remove data older than 90 days"""
        if not self.historical_stats[loan_type]['timestamps']:
            return
            
        cutoff_date = datetime.now() - timedelta(days=90)
        timestamps = [datetime.fromisoformat(ts) for ts in self.historical_stats[loan_type]['timestamps']]
        
        valid_indices = [i for i, ts in enumerate(timestamps) if ts >= cutoff_date]
        
        if valid_indices:
            self.historical_stats[loan_type]['default_rates'] = [
                self.historical_stats[loan_type]['default_rates'][i] for i in valid_indices
            ]
            self.historical_stats[loan_type]['interest_rates'] = [
                self.historical_stats[loan_type]['interest_rates'][i] for i in valid_indices
            ]
            self.historical_stats[loan_type]['loan_amounts'] = [
                self.historical_stats[loan_type]['loan_amounts'][i] for i in valid_indices
            ]
            self.historical_stats[loan_type]['timestamps'] = [
                self.historical_stats[loan_type]['timestamps'][i] for i in valid_indices
            ]

    def get_historical_average(self, loan_type: str, metric: str) -> Tuple[float, float]:
        """Get weighted average from historical data with confidence score"""
        if loan_type not in self.historical_stats:
            return self.get_initial_default(loan_type, metric)
            
        values = self.historical_stats[loan_type][metric]
        if not values:
            return self.get_initial_default(loan_type, metric)
            
        # Calculate weighted average (recent values weighted more heavily)
        weights = np.linspace(0.5, 1.0, len(values))
        weighted_avg = np.average(values, weights=weights)
        
        # Calculate confidence based on number and recency of samples
        max_samples = 30  # Assume 30 samples is optimal
        confidence = min(len(values) / max_samples, 1.0)
        
        # Adjust confidence based on data age
        if self.historical_stats[loan_type]['timestamps']:
            latest_date = datetime.fromisoformat(self.historical_stats[loan_type]['timestamps'][-1])
            days_old = (datetime.now() - latest_date).days
            age_factor = max(0, (90 - days_old) / 90)  # Reduce confidence for older data
            confidence *= age_factor
            
        return weighted_avg, confidence

    def get_initial_default(self, loan_type: str, metric: str) -> Tuple[float, float]:
        """Get initial default values with low confidence"""
        if metric == 'default_rates':
            defaults = {
                'EDUCATION': (10.1, 0.3),  # Federal student loan average
                'MEDICAL': (15.2, 0.3),    # Healthcare lending data
                'VENTURE': (20.0, 0.3),    # SBA loan statistics
                'HOMEIMPROVEMENT': (8.5, 0.3),  # Home improvement loan data
                'DEBTCONSOLIDATION': (12.0, 0.3),  # Consumer credit data
                'PERSONAL': (10.5, 0.3)    # Personal loan statistics
            }
        elif metric == 'interest_rates':
            defaults = {
                'EDUCATION': (7.5, 0.3),   # Federal loan rates
                'MEDICAL': (12.0, 0.3),    # Medical loan averages
                'VENTURE': (15.0, 0.3),    # Business loan rates
                'HOMEIMPROVEMENT': (9.0, 0.3),  # Home improvement rates
                'DEBTCONSOLIDATION': (11.0, 0.3),  # Debt consolidation averages
                'PERSONAL': (10.5, 0.3)    # Personal loan rates
            }
        else:  # loan_amounts
            defaults = {
                'EDUCATION': (35000, 0.3),    # Average student loan
                'MEDICAL': (15000, 0.3),      # Typical medical loan
                'VENTURE': (50000, 0.3),      # Small business average
                'HOMEIMPROVEMENT': (25000, 0.3),  # Home improvement average
                'DEBTCONSOLIDATION': (20000, 0.3),  # Debt consolidation
                'PERSONAL': (15000, 0.3)      # Personal loan average
            }
            
        return defaults.get(loan_type, (0.0, 0.0))

    def get_default_rate(self, loan_type: str) -> float:
        """Get default rate based on historical data"""
        rate, confidence = self.get_historical_average(loan_type, 'default_rates')
        if confidence > 0.5:
            return rate
        return self.get_initial_default(loan_type, 'default_rates')[0]

    def get_default_interest_rate(self, loan_type: str) -> float:
        """Get interest rate based on historical data"""
        rate, confidence = self.get_historical_average(loan_type, 'interest_rates')
        if confidence > 0.5:
            return rate
        return self.get_initial_default(loan_type, 'interest_rates')[0]

    def get_default_loan_amount(self, loan_type: str) -> float:
        """Get loan amount based on historical data"""
        amount, confidence = self.get_historical_average(loan_type, 'loan_amounts')
        if confidence > 0.5:
            return amount
        return self.get_initial_default(loan_type, 'loan_amounts')[0]

    def calculate_loan_statistics(self, loan_stats: Dict[str, List[float]], loan_type: str) -> Dict[str, float]:
        """Calculate statistics with confidence scores and update historical data"""
        result = {}
        
        # Helper function for weighted statistics
        def calculate_weighted_stat(values: List[float], fallback_func, min_samples: int = 3) -> tuple[float, float]:
            if len(values) >= min_samples:
                # Remove outliers using IQR method
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_values = [v for v in values if lower_bound <= v <= upper_bound]
                
                if filtered_values:
                    # Weight recent values more heavily
                    weights = np.linspace(0.5, 1.0, len(filtered_values))
                    weighted_avg = np.average(filtered_values, weights=weights)
                    confidence = min(len(filtered_values) / min_samples, 1.0)
                    return weighted_avg, confidence
                    
            # If not enough valid values, use historical data
            return fallback_func(loan_type), 0.0
        
        # Calculate statistics with confidence scores
        default_rate, default_conf = calculate_weighted_stat(
            loan_stats['default_rates'],
            self.get_default_rate
        )
        
        interest_rate, interest_conf = calculate_weighted_stat(
            loan_stats['interest_rates'],
            self.get_default_interest_rate
        )
        
        loan_amount, amount_conf = calculate_weighted_stat(
            loan_stats['avg_loan_amounts'],
            self.get_default_loan_amount
        )
        
        # Store results with confidence scores
        result = {
            'default_rate': default_rate,
            'default_rate_confidence': default_conf,
            'avg_interest_rate': interest_rate,
            'interest_rate_confidence': interest_conf,
            'avg_loan_amount': loan_amount,
            'loan_amount_confidence': amount_conf
        }
        
        # Update historical statistics
        self.update_historical_stats(loan_type, result)
        self.save_historical_stats()
        
        return result

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