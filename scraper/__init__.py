"""
Web scraping module for image collection.
"""

from .scraper import scrape_images, selenium_scraper, scrape_images_auto

__all__ = ['scrape_images', 'selenium_scraper', 'scrape_images_auto']