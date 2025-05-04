"""
Central configuration file for image collection and similarity system
"""
import os
import torch

# Directory and path settings
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, '_data/_image_corpus')    # Directory containing sample/training images
OUTPUT_DIR = os.path.join(BASE_DIR, '_data/sorted_images')  # Base directory for sorted outputs
SCRAPER_DIR = os.path.join(BASE_DIR, '_data/scraped_images') # Directory for scraped images
MODEL_DIR = os.path.join(BASE_DIR, '_models')               # Directory to store models
MODEL_NAME = 'similarity_model.pkl'                         # Default model filename

# Model parameters
FEATURE_DIMENSIONS = 512                        # Feature vector dimensions from ResNet
SIMILARITY_THRESHOLD = 0.7                      # Default similarity threshold (0.0-1.0)
USE_RECURSIVE_SCAN = True                       # Whether to scan subfolders in sample directory
MODEL_ARCHITECTURE = 'resnet18'                 # Neural network architecture to use (resnet18, resnet34, resnet50)

# Processing parameters
BATCH_SIZE = 32                                 # Batch size for feature extraction
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for computation
NUM_NEIGHBORS = 5                               # Number of nearest neighbors to find
METRIC = 'cosine'                               # Distance metric for similarity ('cosine', 'euclidean', etc.)

# Image parameters
IMAGE_SIZE = (224, 224)                         # Size for images used in feature extraction
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
NORMALIZE_MEAN = [0.485, 0.456, 0.406]          # ImageNet normalization mean
NORMALIZE_STD = [0.229, 0.224, 0.225]           # ImageNet normalization std

# Scraper settings
DEFAULT_NUM_IMAGES = 30                         # Default number of images to scrape
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_TIMEOUT = 15                            # Timeout for web requests in seconds
REQUEST_DELAY_MIN = 0.5                         # Minimum delay between requests
REQUEST_DELAY_MAX = 1.5                         # Maximum delay between requests

# Sorting parameters
DEFAULT_SORT_LEVELS = [0.8, 0.6, 0.4, 0.2]      # Default similarity levels for multi-level sorting
SORT_COPY_BY_DEFAULT = True                     # Whether to copy (True) or move (False) files by default