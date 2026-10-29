# src/data_collection/yelp_collector.py
"""
Yelp Review Scraper with Selenium
Dynamic content requires browser automation
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YelpCollector:
    """Scrape reviews from Yelp using Selenium"""
    
    def __init__(self, rate_limit: float = 2.5, headless: bool = True):
        """
        Initialize Yelp collector
        
        Args:
            rate_limit: Minimum seconds between requests (default 2.5)
            headless: Run browser in headless mode
        """
        self.rate_limit = rate_limit
        self.headless = headless
        self.driver = None
        
    def _init_driver(self):
        """Initialize Chrome WebDriver"""
        if self.driver is not None:
            return
        
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless=new")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Auto-install ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            logger.info("Chrome WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def _close_driver(self):
        """Close Chrome WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("Chrome WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")
    
    def collect_reviews(self, business_url: str, limit: int = 60) -> List[Dict]:
        """
        Collect reviews from a Yelp business page
        
        Args:
            business_url: Yelp business URL (e.g., "https://www.yelp.com/biz/business-name")
            limit: Maximum number of reviews to collect
            
        Returns:
            List of review dictionaries
        """
        logger.info(f"Starting Yelp collection from: {business_url}")
        
        reviews = []
        
        try:
            # Initialize WebDriver
            self._init_driver()
            
            # Navigate to business page
            self.driver.get(business_url)
            time.sleep(3)  # Wait for initial page load
            
            # Extract business name from page
            try:
                business_name_elem = self.driver.find_element(By.CSS_SELECTOR, "h1")
                business_name = business_name_elem.text
                logger.info(f"Business: {business_name}")
            except:
                business_name = "Unknown Business"
            
            # Scroll to load more reviews
            scroll_pause = 2
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while len(reviews) < limit:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause)
                
                # Get page source and parse
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                # Find review elements (Yelp structure as of 2025)
                # Try multiple selectors as Yelp frequently updates their HTML
                review_elements = []
                
                # Try data-testid selector
                review_elements = soup.find_all('li', {'data-testid': lambda x: x and 'review' in str(x).lower()})
                
                # Try section with review content
                if not review_elements:
                    review_elements = soup.find_all('section', {'aria-label': lambda x: x and 'review' in str(x).lower()})
                
                # Try div with review class patterns
                if not review_elements:
                    review_elements = soup.find_all('div', class_=lambda x: x and 'review' in str(x).lower() and 'container' in str(x).lower())
                
                # Fallback: find ul with reviews and get li children
                if not review_elements:
                    review_list = soup.find('ul', class_=lambda x: x and 'list' in str(x).lower())
                    if review_list:
                        review_elements = review_list.find_all('li', recursive=False)
                
                logger.info(f"Found {len(review_elements)} review elements on page")
                
                # Parse reviews
                for elem in review_elements:
                    if len(reviews) >= limit:
                        break
                    
                    try:
                        review_data = self._parse_review_element(elem, business_name)
                        if review_data and review_data not in reviews:
                            reviews.append(review_data)
                            logger.debug(f"Collected review {len(reviews)}/{limit}")
                    except Exception as e:
                        logger.error(f"Error parsing review: {e}")
                        continue
                
                # Check if we've reached the bottom or loaded enough reviews
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height or len(reviews) >= limit:
                    break
                
                last_height = new_height
                time.sleep(self.rate_limit)
            
        except Exception as e:
            logger.error(f"Error during Yelp scraping: {e}", exc_info=True)
        
        finally:
            self._close_driver()
        
        logger.info(f"Collected {len(reviews)} reviews from Yelp")
        return reviews[:limit]
    
    def _parse_review_element(self, elem, business_name: str) -> Dict:
        """
        Parse a single review element
        
        Args:
            elem: BeautifulSoup element
            business_name: Name of the business
            
        Returns:
            Dictionary with review data
        """
        try:
            # Extract rating (look for aria-label with "star rating")
            rating = 0.0
            rating_elem = elem.find('div', {'role': 'img', 'aria-label': lambda x: x and 'star' in str(x).lower()})
            if not rating_elem:
                rating_elem = elem.find(attrs={'aria-label': lambda x: x and 'star rating' in str(x).lower()})
            
            if rating_elem and 'aria-label' in rating_elem.attrs:
                rating_text = rating_elem['aria-label']
                # Parse "X star rating" or "X.5 star rating"
                try:
                    rating = float(rating_text.split()[0])
                except:
                    rating = 0.0
            
            # Extract review text - try multiple selectors
            review_text = ""
            
            # Try span with raw text
            review_text_elem = elem.find('span', {'lang': True})
            if review_text_elem:
                review_text = review_text_elem.get_text(strip=True)
            
            # Try p tag with comment
            if not review_text:
                review_text_elem = elem.find('p', class_=lambda x: x and 'comment' in str(x).lower())
                if review_text_elem:
                    review_text = review_text_elem.get_text(strip=True)
            
            # Try any p tag in the element
            if not review_text:
                review_text_elem = elem.find('p')
                if review_text_elem:
                    review_text = review_text_elem.get_text(strip=True)
            
            # Try span with long text
            if not review_text:
                spans = elem.find_all('span')
                for span in spans:
                    text = span.get_text(strip=True)
                    if len(text) > 50:  # Assume review text is longer than 50 chars
                        review_text = text
                        break
            
            if not review_text:
                return None
            
            # Extract reviewer name
            reviewer_name = "Anonymous"
            reviewer_elem = elem.find('a', attrs={'href': lambda x: x and '/user_details' in str(x)})
            if reviewer_elem:
                reviewer_name = reviewer_elem.get_text(strip=True)
            
            # Extract date
            review_date = datetime.now().strftime("%Y-%m-%d")
            date_elem = elem.find('span', class_=lambda x: x and 'date' in str(x).lower() if x else False)
            if date_elem:
                review_date = date_elem.get_text(strip=True)
            
            return {
                'platform': 'yelp',
                'rating': rating,
                'review_text': review_text,
                'review_date': review_date,
                'product_name': business_name,
                'category': 'restaurant',  # Default category
                'reviewer_name': reviewer_name,
                'verified_purchase': False,  # Yelp doesn't have this
                'word_count': len(review_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error parsing review element: {e}")
            return None


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Yelp Collector Test")
    print("=" * 60)
    
    collector = YelpCollector(headless=False)  # Show browser for testing
    
    test_url = "https://www.yelp.com/biz/food-for-friends-brighton"
    
    print(f"\nTesting URL: {test_url}")
    print("Collecting 5 reviews (this may take 30-60 seconds)...")
    
    reviews = collector.collect_reviews(test_url, limit=5)
    
    print(f"\nCollected {len(reviews)} reviews:")
    for i, review in enumerate(reviews, 1):
        print(f"\n{i}. Rating: {review['rating']} | Reviewer: {review['reviewer_name']}")
        print(f"   Text: {review['review_text'][:100]}...")
    
    print("\n" + "=" * 60)
