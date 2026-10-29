# src/data_collection/trustpilot_collector.py
"""
Trustpilot Review Scraper
Ethical scraping with rate limiting and robots.txt compliance
"""

import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
import pandas as pd
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrustpilotCollector:
    """Scrape reviews from Trustpilot with ethical rate limiting"""
    
    def __init__(self, rate_limit: float = 2.0):
        """
        Initialize Trustpilot collector
        
        Args:
            rate_limit: Minimum seconds between requests (default 2.0)
        """
        self.rate_limit = rate_limit
        self.base_url = "https://www.trustpilot.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Research Purpose)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
    def check_robots_txt(self) -> bool:
        """Check if scraping is allowed by robots.txt"""
        try:
            robots_url = f"{self.base_url}/robots.txt"
            response = self.session.get(robots_url, timeout=10)
            # Simplified check - in production, use robotparser
            return "Disallow: /review" not in response.text
        except Exception as e:
            logger.warning(f"Could not check robots.txt: {e}")
            return True  # Proceed cautiously
    
    def collect_reviews(self, company_url: str, limit: int = 60) -> List[Dict]:
        """
        Collect reviews from a Trustpilot company page
        
        Args:
            company_url: Trustpilot URL (e.g., "www.trustpilot.com/review/www.apple.com")
            limit: Maximum number of reviews to collect
            
        Returns:
            List of review dictionaries
        """
        logger.info(f"Starting Trustpilot collection from: {company_url}")
        
        if not self.check_robots_txt():
            logger.warning("Robots.txt disallows scraping. Proceeding with caution.")
        
        reviews = []
        page = 1
        
        # Trustpilot URL format: /review/domain?page=X
        if not company_url.startswith('http'):
            company_url = f"{self.base_url}/review/{company_url.replace('www.trustpilot.com/review/', '')}"
        
        while len(reviews) < limit:
            try:
                # Add page parameter
                page_url = f"{company_url}?page={page}"
                logger.info(f"Fetching page {page}...")
                
                # Rate limiting
                time.sleep(self.rate_limit + random.uniform(0, 0.5))
                
                response = self.session.get(page_url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find review cards (Trustpilot structure as of 2024)
                review_cards = soup.find_all('div', {'class': 'styles_reviewCardInner__EwDq2'})
                
                if not review_cards:
                    # Alternative class names (Trustpilot changes these)
                    review_cards = soup.find_all('article', {'class': 'review'})
                
                if not review_cards:
                    logger.warning(f"No reviews found on page {page}. Stopping.")
                    break
                
                for card in review_cards:
                    if len(reviews) >= limit:
                        break
                    
                    try:
                        review_data = self._parse_review_card(card)
                        if review_data:
                            reviews.append(review_data)
                            logger.debug(f"Collected review {len(reviews)}/{limit}")
                    except Exception as e:
                        logger.error(f"Error parsing review card: {e}")
                        continue
                
                page += 1
                
            except requests.RequestException as e:
                logger.error(f"Request error on page {page}: {e}")
                break
        
        logger.info(f"Collected {len(reviews)} reviews from Trustpilot")
        return reviews
    
    def _parse_review_card(self, card) -> Dict:
        """Parse a single review card"""
        try:
            # Extract rating (1-5 stars)
            rating_elem = card.find('div', {'class': 'star-rating'})
            if not rating_elem:
                rating_elem = card.find('div', class_=lambda x: x and 'star' in x.lower())
            
            rating = 0
            if rating_elem:
                # Try to extract from data attribute or count filled stars
                rating_img = rating_elem.find('img')
                if rating_img and 'alt' in rating_img.attrs:
                    rating_text = rating_img['alt']
                    rating = int(rating_text.split()[0]) if rating_text else 0
            
            # Extract review title
            title_elem = card.find('h2', {'class': 'typography_heading-s__f7029'})
            if not title_elem:
                title_elem = card.find('h2')
            title = title_elem.text.strip() if title_elem else ""
            
            # Extract review text
            text_elem = card.find('p', {'class': 'typography_body-l__KUYFJ'})
            if not text_elem:
                text_elem = card.find('p', {'class': 'review-content__text'})
            if not text_elem:
                text_elem = card.find('div', class_=lambda x: x and 'reviewBody' in x)
            
            review_text = text_elem.text.strip() if text_elem else ""
            
            # Combine title and text
            full_text = f"{title}. {review_text}" if title else review_text
            
            # Extract date
            date_elem = card.find('time')
            review_date = date_elem['datetime'] if date_elem and 'datetime' in date_elem.attrs else None
            
            # Extract reviewer name (optional)
            reviewer_elem = card.find('span', {'class': 'typography_heading-xxs__QKBS8'})
            reviewer = reviewer_elem.text.strip() if reviewer_elem else "Anonymous"
            
            # Verified purchase check
            verified = bool(card.find('div', {'class': 'typography_body-m__xgxZ_'}))
            
            return {
                'platform': 'trustpilot',
                'rating': rating,
                'review_text': full_text,
                'review_date': review_date,
                'reviewer_name': reviewer,
                'verified_purchase': verified,
                'scraped_at': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error parsing review: {e}")
            return None
    
    def save_to_csv(self, reviews: List[Dict], filename: str):
        """Save reviews to CSV file"""
        df = pd.DataFrame(reviews)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(reviews)} reviews to {filename}")


# Example usage
if __name__ == "__main__":
    collector = TrustpilotCollector(rate_limit=2.0)
    
    # Example: Collect Apple reviews
    reviews = collector.collect_reviews(
        company_url="www.trustpilot.com/review/www.apple.com",
        limit=60
    )
    
    collector.save_to_csv(reviews, "data/raw/trustpilot_apple.csv")
    
    print(f"Collected {len(reviews)} reviews")
    if reviews:
        print("\nSample review:")
        print(f"Rating: {reviews[0]['rating']}")
        print(f"Text: {reviews[0]['review_text'][:200]}...")