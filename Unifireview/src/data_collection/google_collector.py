# src/data_collection/google_collector.py
"""
Google Reviews Collector with Apify API and Selenium fallback
"""

from apify_client import ApifyClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import requests
import time
import random
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleReviewsCollector:
    """Collect Google Reviews using Apify API or Selenium"""
    
    def __init__(self, apify_key: Optional[str] = None, use_selenium: bool = False):
        """
        Initialize Google Reviews collector
        
        Args:
            apify_key: Apify API key for Google Maps scraping
            use_selenium: Whether to use Selenium as default method
        """
        self.apify_key = apify_key
        self.use_selenium = use_selenium
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        })
    
    def collect_reviews_apify(self, place_url: str, limit: int = 60, api_key: Optional[str] = None) -> List[Dict]:
        """
        Collect reviews using Apify API (Primary method)
        
        Args:
            place_url: Google Maps place URL (full URL or shortened goo.gl link)
            limit: Number of reviews to collect
            api_key: Apify API key (uses self.apify_key if not provided)
            
        Returns:
            List of review dictionaries
        """
        api_key = api_key or self.apify_key
        
        if not api_key:
            logger.error("No Apify API key provided. Cannot use Apify method.")
            return []
        
        try:
            logger.info(f"Starting Apify scraping for: {place_url}")
            
            # Initialize Apify client
            client = ApifyClient(api_key)
            
            # Prepare the Actor input
            run_input = {
                "startUrls": [{"url": place_url}],
                "maxReviews": limit,
                "reviewsSort": "newest",
                "language": "en",
            }
            
            # Run the Actor (Google Maps Scraper)
            # Actor ID: compass/google-maps-scraper (may not exist - will use selenium fallback)
            logger.info("Running Apify Google Maps Scraper...")
            run = client.actor("compass/google-maps-scraper").call(run_input=run_input)
            
            # Fetch results
            reviews = []
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                # Extract reviews from the place data
                place_reviews = item.get('reviews', [])
                
                for review in place_reviews[:limit]:
                    reviews.append({
                        'platform': 'google',
                        'rating': review.get('stars', 0.0),
                        'review_text': review.get('text', ''),
                        'review_date': review.get('publishedAtDate', datetime.now().strftime("%Y-%m-%d")),
                        'product_name': item.get('title', 'Unknown Place'),
                        'category': item.get('categoryName', 'Unknown'),
                        'reviewer_name': review.get('name', 'Anonymous'),
                        'verified_purchase': review.get('isLocalGuide', False),
                        'word_count': len(review.get('text', '').split()),
                        'scraped_at': datetime.now().isoformat()
                    })
                
                if len(reviews) >= limit:
                    break
            
            logger.info(f"Collected {len(reviews)} reviews via Apify")
            return reviews[:limit]
            
        except Exception as e:
            logger.error(f"Apify scraping error: {e}", exc_info=True)
            return []
    
    def collect_reviews_selenium(self, place_url: str, limit: int = 60, headless: bool = True) -> List[Dict]:
        """
        Collect reviews using Selenium (Fallback method)
        
        Args:
            place_url: Google Maps place URL
            limit: Number of reviews to collect
            headless: Run browser in headless mode
            
        Returns:
            List of review dictionaries
        """
        logger.info(f"Starting Selenium scraping for: {place_url}")
        
        reviews = []
        
        try:
            # Initialize WebDriver
            self._init_driver(headless=headless)
            
            # Navigate to place
            self.driver.get(place_url)
            time.sleep(5)  # Wait for page load
            
            # Click on Reviews tab
            try:
                reviews_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'Reviews')]"))
                )
                reviews_button.click()
                time.sleep(3)
            except:
                logger.warning("Could not find Reviews tab, continuing anyway...")
            
            # Scroll to load more reviews
            scrollable_div = self.driver.find_element(By.XPATH, "//div[@role='feed']")
            
            for _ in range(10):  # Scroll 10 times
                self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
                time.sleep(2)
            
            # Parse reviews
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            review_elements = soup.find_all('div', {'data-review-id': True})
            
            logger.info(f"Found {len(review_elements)} review elements")
            
            for elem in review_elements[:limit]:
                try:
                    review_data = self._parse_google_review(elem)
                    if review_data:
                        reviews.append(review_data)
                except Exception as e:
                    logger.error(f"Error parsing review: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Selenium scraping error: {e}", exc_info=True)
        
        finally:
            self._close_driver()
        
        logger.info(f"Collected {len(reviews)} reviews via Selenium")
        return reviews
    
    def _init_driver(self, headless: bool = True):
        """Initialize Chrome WebDriver"""
        if self.driver is not None:
            return
        
        try:
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument("--headless=new")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            # Auto-install ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            logger.info("Chrome WebDriver initialized")
            
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
    
    def _parse_google_review(self, elem) -> Optional[Dict]:
        """Parse a Google review element"""
        try:
            # Extract rating
            rating_elem = elem.find('span', {'role': 'img'})
            rating = 0.0
            if rating_elem and 'aria-label' in rating_elem.attrs:
                rating_text = rating_elem['aria-label']
                if 'stars' in rating_text.lower():
                    rating = float(rating_text.split()[0])
            
            # Extract text
            text_elem = elem.find('span', class_=lambda x: x and 'review-full-text' in x.lower() if x else False)
            if not text_elem:
                text_elem = elem.find('span', {'data-expandable-section': True})
            
            review_text = text_elem.get_text(strip=True) if text_elem else ""
            
            if not review_text:
                return None
            
            # Extract reviewer name
            name_elem = elem.find('div', class_=lambda x: x and 'name' in x.lower() if x else False)
            reviewer_name = name_elem.get_text(strip=True) if name_elem else "Anonymous"
            
            # Extract date
            date_elem = elem.find('span', class_=lambda x: x and 'date' in x.lower() if x else False)
            review_date = date_elem.get_text(strip=True) if date_elem else datetime.now().strftime("%Y-%m-%d")
            
            return {
                'platform': 'google',
                'rating': rating,
                'review_text': review_text,
                'review_date': review_date,
                'product_name': 'Google Maps Location',
                'category': 'location',
                'reviewer_name': reviewer_name,
                'verified_purchase': False,
                'word_count': len(review_text.split()),
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing Google review: {e}")
            return None
    
    def collect_reviews(self, place_url: str, limit: int = 60, method: str = 'apify') -> List[Dict]:
        """
        Main method to collect reviews (auto-selects method)
        
        Args:
            place_url: Google Maps URL
            limit: Number of reviews
            method: 'apify' or 'selenium'
            
        Returns:
            List of reviews
        """
        if method == 'apify' and self.apify_key:
            return self.collect_reviews_apify(place_url, limit)
        else:
            logger.info("Using Selenium fallback")
            return self.collect_reviews_selenium(place_url, limit)


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Google Reviews Collector Test")
    print("=" * 60)
    
    # Test with Apify
    apify_key = os.getenv("APIFY_API_KEY")
    collector = GoogleReviewsCollector(apify_key=apify_key)
    
    test_url = "https://maps.app.goo.gl/Fyjq1LtEM6JZEHMg6"
    
    print(f"\nTest URL: {test_url}")
    print("Collecting 5 reviews via Apify...")
    
    reviews = collector.collect_reviews(test_url, limit=5, method='apify')
    
    print(f"\nCollected {len(reviews)} reviews:")
    for i, review in enumerate(reviews, 1):
        print(f"\n{i}. Rating: {review['rating']} | Reviewer: {review['reviewer_name']}")
        print(f"   Text: {review['review_text'][:100]}...")
    
    print("\n" + "=" * 60)
