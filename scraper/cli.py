#!/usr/bin/env python3
"""
Command-line interface for the Image Scraper
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scraper_cli')

# Make sure imports work regardless of how the script is run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our implementation
from scraper.scraper import scrape_images, selenium_scraper, scrape_images_auto
from utils import config

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Web Image Scraper CLI")
    
    parser.add_argument("search_term", help="Search term for images")
    parser.add_argument("--output", "-o", default=config.SCRAPER_DIR, 
                     help=f"Output directory (default: {config.SCRAPER_DIR})")
    parser.add_argument("--num", "-n", type=int, default=config.DEFAULT_NUM_IMAGES, 
                     help=f"Number of images to download (default: {config.DEFAULT_NUM_IMAGES})")
    parser.add_argument("--method", "-m", choices=['auto', 'selenium', 'standard'], default='auto',
                     help="Scraping method to use (default: auto)")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Scraping '{args.search_term}' for {args.num} images using {args.method} method")
    
    # Choose the appropriate scraper based on the specified method
    try:
        if args.method == 'selenium':
            logger.info("Using Selenium-based scraper")
            # Try the selenium scraper with fallback
            try:
                downloaded = selenium_scraper(args.search_term, args.output, args.num)
            except Exception as e:
                logger.error(f"Selenium scraper failed: {e}")
                logger.info("Falling back to standard scraper")
                downloaded = scrape_images(args.search_term, args.output, args.num)
        elif args.method == 'standard':
            logger.info("Using standard scraper")
            downloaded = scrape_images(args.search_term, args.output, args.num)
        else:  # 'auto'
            logger.info("Automatically selecting best scraper")
            downloaded = scrape_images_auto(args.search_term, args.output, args.num)
        
        if downloaded > 0:
            logger.info(f"Successfully downloaded {downloaded} images to {args.output}")
            return 0
        else:
            logger.error("Failed to download any images")
            return 1
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())