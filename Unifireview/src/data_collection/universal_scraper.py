"""
Universal Scraper
Automatically routes URLs to the appropriate platform scraper
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.url_detector import URLDetector, detect_platform, extract_metadata
from src.data_collection.trustpilot_collector import TrustpilotCollector
from src.data_collection.google_collector import GoogleReviewsCollector
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalScraper:
    """
    Universal review scraper that automatically detects platform and routes to appropriate scraper
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Universal Scraper
        
        Args:
            config: Optional configuration dictionary with scraping settings
        """
        self.config = config or {}
        self.url_detector = URLDetector()
        
        # Initialize platform scrapers
        self.trustpilot_scraper = None
        self.yelp_scraper = None
        self.google_scraper = None
        
    def _init_trustpilot(self):
        """Lazy init Trustpilot scraper"""
        if self.trustpilot_scraper is None:
            rate_limit = self.config.get('trustpilot', {}).get('rate_limit', 2.0)
            self.trustpilot_scraper = TrustpilotCollector(rate_limit=rate_limit)
        return self.trustpilot_scraper
    
    def _init_yelp(self):
        """Lazy init Yelp scraper"""
        if self.yelp_scraper is None:
            try:
                from src.data_collection.yelp_collector import YelpCollector
                rate_limit = self.config.get('yelp', {}).get('rate_limit', 2.5)
                self.yelp_scraper = YelpCollector(rate_limit=rate_limit)
            except ImportError:
                logger.warning("YelpCollector not found. Will be created.")
                self.yelp_scraper = None
        return self.yelp_scraper
    
    def _init_google(self):
        """Lazy init Google scraper"""
        if self.google_scraper is None:
            self.google_scraper = GoogleReviewsCollector()
        return self.google_scraper
    
    def scrape_url(self, url: str, limit: int = 60) -> Dict:
        """
        Scrape reviews from any supported URL
        
        Args:
            url: URL of the review page (Trustpilot, Yelp, or Google Maps)
            limit: Maximum number of reviews to collect
            
        Returns:
            Dictionary with:
                - platform: Platform name
                - url: Original URL
                - reviews: List of review dictionaries
                - metadata: Extracted metadata (company name, etc.)
                - timestamp: Scraping timestamp
                - success: Boolean indicating success
                - error: Error message if failed
        """
        logger.info(f"Starting universal scraping for: {url}")
        
        # Detect platform
        platform = detect_platform(url)
        metadata = extract_metadata(url)
        
        result = {
            'platform': platform,
            'url': url,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'reviews': [],
            'success': False,
            'error': None,
        }
        
        if platform == 'unknown':
            result['error'] = f"Unsupported platform. Supported: {self.url_detector.get_supported_platforms()}"
            logger.error(result['error'])
            return result
        
        # Route to appropriate scraper
        try:
            if platform == 'trustpilot':
                scraper = self._init_trustpilot()
                company_name = metadata.get('company_name', '')
                if company_name:
                    reviews = scraper.collect_reviews(company_url=url, limit=limit)
                    result['reviews'] = reviews
                    result['success'] = True
                else:
                    result['error'] = "Could not extract company name from Trustpilot URL"
            
            elif platform == 'yelp':
                scraper = self._init_yelp()
                if scraper is None:
                    result['error'] = "Yelp scraper not yet implemented"
                    logger.warning(result['error'])
                else:
                    business_id = metadata.get('business_id', '')
                    if business_id:
                        reviews = scraper.collect_reviews(business_url=url, limit=limit)
                        result['reviews'] = reviews
                        result['success'] = True
                    else:
                        result['error'] = "Could not extract business ID from Yelp URL"
            
            elif platform == 'google':
                scraper = self._init_google()
                place_id = metadata.get('place_id', '')
                if place_id:
                    # Try Apify method if configured, fallback to Selenium
                    apify_key = self.config.get('google', {}).get('apify_key')
                    reviews = []
                    
                    if apify_key:
                        try:
                            reviews = scraper.collect_reviews_apify(place_url=url, limit=limit, api_key=apify_key)
                        except Exception as e:
                            logger.warning(f"Apify failed, falling back to Selenium: {e}")
                    
                    # If no reviews from Apify or no Apify key, use Selenium
                    if not reviews:
                        logger.info("Using Selenium fallback for Google Maps scraping")
                        reviews = scraper.collect_reviews_selenium(place_url=url, limit=limit)
                    
                    result['reviews'] = reviews
                    result['success'] = True
                else:
                    result['error'] = "Could not extract place ID from Google Maps URL"
        
        except Exception as e:
            result['error'] = f"Scraping error: {str(e)}"
            logger.error(f"Error scraping {platform}: {e}", exc_info=True)
        
        if result['success']:
            logger.info(f"Successfully scraped {len(result['reviews'])} reviews from {platform}")
        
        return result
    
    def scrape_to_dataframe(self, url: str, limit: int = 60) -> pd.DataFrame:
        """
        Scrape reviews and return as DataFrame
        
        Args:
            url: URL of the review page
            limit: Maximum number of reviews
            
        Returns:
            DataFrame with reviews
        """
        result = self.scrape_url(url, limit)
        
        if not result['success']:
            logger.error(f"Scraping failed: {result['error']}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result['reviews'])
        
        # Add metadata columns
        df['source_url'] = result['url']
        df['scraped_at'] = result['timestamp']
        
        return df
    
    def scrape_to_csv(self, url: str, output_path: str, limit: int = 60) -> bool:
        """
        Scrape reviews and save to CSV
        
        Args:
            url: URL of the review page
            output_path: Path to save CSV file
            limit: Maximum number of reviews
            
        Returns:
            True if successful, False otherwise
        """
        df = self.scrape_to_dataframe(url, limit)
        
        if df.empty:
            logger.error("No data to save")
            return False
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} reviews to {output_path}")
        
        return True
    
    def validate_url(self, url: str) -> bool:
        """
        Check if URL is from a supported platform
        
        Args:
            url: URL to validate
            
        Returns:
            True if supported, False otherwise
        """
        return self.url_detector.validate_url(url)
    
    def get_platform(self, url: str) -> str:
        """
        Get platform name from URL
        
        Args:
            url: URL to analyze
            
        Returns:
            Platform name or 'unknown'
        """
        return self.url_detector.detect_platform(url)


# Convenience function
def scrape_reviews(url: str, limit: int = 60, config: Optional[Dict] = None) -> Dict:
    """
    Scrape reviews from any supported URL
    
    Args:
        url: URL of the review page
        limit: Maximum number of reviews
        config: Optional configuration dictionary
        
    Returns:
        Result dictionary with reviews and metadata
    """
    scraper = UniversalScraper(config=config)
    return scraper.scrape_url(url, limit)


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Universal Scraper Test")
    print("=" * 60)
    
    # Test URLs
    test_urls = [
        "https://www.trustpilot.com/review/www.asus.com",
        "https://www.yelp.com/biz/food-for-friends-brighton",
        "https://maps.app.goo.gl/Fyjq1LtEM6JZEHMg6",
    ]
    
    scraper = UniversalScraper()
    
    for url in test_urls:
        print(f"\n{'-' * 60}")
        print(f"URL: {url}")
        platform = scraper.get_platform(url)
        valid = scraper.validate_url(url)
        
        print(f"Platform: {platform}")
        print(f"Valid: {valid}")
        
        if valid:
            print(f"Ready to scrape (use: scraper.scrape_url('{url}', limit=10))")
    
    print(f"\n{'=' * 60}")
