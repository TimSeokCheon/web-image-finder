"""
Model module for image feature extraction and similarity classification.
"""

from .feature_extractor import extract_features, get_all_images_from_folder
from .classifier import ImageSimilarityClassifier

__all__ = ['extract_features', 'get_all_images_from_folder', 'ImageSimilarityClassifier']