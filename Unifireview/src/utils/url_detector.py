"""
URL Detection Utility
Automatically detects the platform (Trustpilot, Yelp, Google Maps) from a given URL
"""

from urllib.parse import urlparse
import re
from typing import Optional, Dict


class URLDetector:
    """Detects platform from URLs"""
    
    PLATFORM_PATTERNS = {
        'trustpilot': [
            r'trustpilot\.com',
        ],
        'yelp': [
            r'yelp\.com',
            r'yelp\.[a-z]{2,}',  # yelp.co.uk, yelp.ca, etc.
        ],
        'google': [
            r'google\.com/maps',
            r'maps\.google\.com',
            r'maps\.app\.goo\.gl',
            r'goo\.gl/maps',
        ]
    }
    
    @classmethod
    def detect_platform(cls, url: str) -> str:
        """
        Detect the platform from a URL
        
        Args:
            url: The URL to analyze
            
        Returns:
            Platform name: 'trustpilot', 'yelp', 'google', or 'unknown'
        """
        if not url:
            return 'unknown'
        
        # Normalize URL
        url = url.lower().strip()
        
        # Parse URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path  # Handle cases like "trustpilot.com" without https://
        except Exception:
            return 'unknown'
        
        # Check against patterns
        for platform, patterns in cls.PLATFORM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, url):
                    return platform
        
        return 'unknown'
    
    @classmethod
    def extract_metadata(cls, url: str) -> Dict[str, str]:
        """
        Extract metadata from URL (platform, company/business name, etc.)
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary with platform and extracted metadata
        """
        platform = cls.detect_platform(url)
        metadata = {
            'platform': platform,
            'url': url,
            'company_name': None,
            'business_id': None,
        }
        
        if platform == 'trustpilot':
            # Extract company name from Trustpilot URL
            # Pattern: trustpilot.com/review/COMPANY_NAME
            match = re.search(r'trustpilot\.com/review/([^/?&#]+)', url)
            if match:
                metadata['company_name'] = match.group(1)
        
        elif platform == 'yelp':
            # Extract business ID from Yelp URL
            # Pattern: yelp.com/biz/BUSINESS_ID
            match = re.search(r'yelp\.[a-z.]+/biz/([^/?&#]+)', url)
            if match:
                metadata['business_id'] = match.group(1)
        
        elif platform == 'google':
            # Google Maps URLs are complex, store full URL
            # Pattern: maps.app.goo.gl/PLACE_ID or google.com/maps/place/...
            if 'maps.app.goo.gl' in url or 'goo.gl/maps' in url:
                # Shortened URL - extract the code
                match = re.search(r'(?:maps\.app\.)?goo\.gl/(?:maps/)?([A-Za-z0-9]+)', url)
                if match:
                    metadata['place_id'] = match.group(1)
            else:
                # Try to extract place name from full URL
                match = re.search(r'/place/([^/?&#]+)', url)
                if match:
                    metadata['place_name'] = match.group(1).replace('+', ' ')
        
        return metadata
    
    @classmethod
    def validate_url(cls, url: str) -> bool:
        """
        Validate if URL is from a supported platform
        
        Args:
            url: The URL to validate
            
        Returns:
            True if platform is supported, False otherwise
        """
        platform = cls.detect_platform(url)
        return platform != 'unknown'
    
    @classmethod
    def get_supported_platforms(cls) -> list:
        """
        Get list of supported platforms
        
        Returns:
            List of platform names
        """
        return list(cls.PLATFORM_PATTERNS.keys())


def detect_platform(url: str) -> str:
    """
    Convenience function to detect platform from URL
    
    Args:
        url: The URL to analyze
        
    Returns:
        Platform name: 'trustpilot', 'yelp', 'google', or 'unknown'
    """
    return URLDetector.detect_platform(url)


def extract_metadata(url: str) -> Dict[str, str]:
    """
    Convenience function to extract metadata from URL
    
    Args:
        url: The URL to analyze
        
    Returns:
        Dictionary with platform and extracted metadata
    """
    return URLDetector.extract_metadata(url)


# Test function
if __name__ == "__main__":
    # Test URLs
    test_urls = [
        "https://www.trustpilot.com/review/www.asus.com",
        "https://www.yelp.com/biz/food-for-friends-brighton",
        "https://maps.app.goo.gl/Fyjq1LtEM6JZEHMg6",
        "https://www.trustpilot.com/review/netflix.com",
        "https://www.amazon.com/product",  # Should be unknown
    ]
    
    print("=" * 60)
    print("URL Detector Test")
    print("=" * 60)
    
    for url in test_urls:
        print(f"\nURL: {url}")
        platform = detect_platform(url)
        metadata = extract_metadata(url)
        
        print(f"  Platform: {platform}")
        print(f"  Valid: {URLDetector.validate_url(url)}")
        print(f"  Metadata: {metadata}")
    
    print("\n" + "=" * 60)
    print(f"Supported platforms: {URLDetector.get_supported_platforms()}")
    print("=" * 60)
