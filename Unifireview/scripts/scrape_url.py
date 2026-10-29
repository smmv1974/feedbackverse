"""
CLI Tool for URL-based Review Scraping
Usage: python scrape_url.py --url <URL> --limit <NUMBER>
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collection.universal_scraper import UniversalScraper
from src.utils.url_detector import detect_platform, extract_metadata
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config.get('scraping', {})
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        return {}


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Universal Review Scraper - Scrape reviews from Trustpilot, Yelp, or Google Maps"
    )
    
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='URL of the review page (Trustpilot, Yelp, or Google Maps)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=60,
        help='Maximum number of reviews to collect (default: 60)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated in data/scraped/)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json', 'both'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Display header
    print("=" * 70)
    print("🔗 UNIVERSAL REVIEW SCRAPER")
    print("=" * 70)
    
    # Detect platform
    platform = detect_platform(args.url)
    metadata = extract_metadata(args.url)
    
    print(f"\n📍 URL: {args.url}")
    print(f"🏷️  Platform: {platform.upper()}")
    
    if platform == 'unknown':
        print("\n❌ ERROR: Unsupported platform!")
        print("✅ Supported platforms: Trustpilot, Yelp, Google Maps")
        sys.exit(1)
    
    # Display metadata
    if metadata.get('company_name'):
        print(f"🏢 Company: {metadata['company_name']}")
    if metadata.get('business_id'):
        print(f"🏪 Business: {metadata['business_id'].replace('-', ' ').title()}")
    if metadata.get('place_id'):
        print(f"📌 Place ID: {metadata['place_id']}")
    
    print(f"🎯 Target: {args.limit} reviews")
    print(f"📄 Format: {args.format.upper()}")
    
    # Load config
    config = load_config(args.config)
    
    # Initialize scraper
    print("\n" + "-" * 70)
    print("🚀 Starting scraper...")
    print("-" * 70 + "\n")
    
    scraper = UniversalScraper(config=config)
    
    # Scrape reviews
    result = scraper.scrape_url(args.url, limit=args.limit)
    
    if not result['success']:
        print(f"\n❌ Scraping failed: {result['error']}")
        sys.exit(1)
    
    reviews = result['reviews']
    
    # Display summary
    print("\n" + "=" * 70)
    print("✅ SCRAPING COMPLETE!")
    print("=" * 70)
    print(f"📊 Collected: {len(reviews)} reviews")
    
    if reviews:
        avg_rating = sum(r.get('rating', 0) for r in reviews) / len(reviews)
        print(f"⭐ Average Rating: {avg_rating:.2f}")
        
        # Calculate rating distribution
        ratings = {}
        for r in reviews:
            rating = int(r.get('rating', 0))
            ratings[rating] = ratings.get(rating, 0) + 1
        
        print("\n📈 Rating Distribution:")
        for rating in sorted(ratings.keys(), reverse=True):
            bar = "█" * int(ratings[rating] / len(reviews) * 50)
            print(f"  {rating}⭐: {bar} ({ratings[rating]})")
    
    # Save to file
    if args.output:
        output_path = args.output
    else:
        # Auto-generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company = metadata.get('company_name') or metadata.get('business_id') or metadata.get('place_id') or 'reviews'
        company = company.replace('www.', '').replace('.com', '').replace('-', '_')
        
        output_dir = Path("data/scraped")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'both':
            output_csv = output_dir / f"{platform}_{company}_{timestamp}.csv"
            output_json = output_dir / f"{platform}_{company}_{timestamp}.json"
            output_path = str(output_csv)
        elif args.format == 'json':
            output_path = str(output_dir / f"{platform}_{company}_{timestamp}.json")
        else:
            output_path = str(output_dir / f"{platform}_{company}_{timestamp}.csv")
    
    # Save
    print(f"\n💾 Saving to: {output_path}")
    
    if args.format == 'csv' or args.format == 'both':
        import pandas as pd
        df = pd.DataFrame(reviews)
        csv_path = output_path if output_path.endswith('.csv') else output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV saved: {csv_path}")
    
    if args.format == 'json' or args.format == 'both':
        import json
        json_path = output_path if output_path.endswith('.json') else output_path.replace('.csv', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved: {json_path}")
    
    print("\n" + "=" * 70)
    print("🎉 Done! Reviews ready for LLM analysis.")
    print("=" * 70)
    print(f"\n💡 Next step: python test_analysis.py")
    print()


if __name__ == "__main__":
    main()
