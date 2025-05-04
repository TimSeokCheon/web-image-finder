"""
Image Similarity Collection Project

A structured toolkit for collecting, comparing, and organizing images based on visual similarity.
This project contains tools for web scraping, image processing, and similarity-based classification.
"""

from . import scraper
from . import model
from . import utils

__version__ = "1.0.0"
__all__ = ['scraper', 'model', 'utils']