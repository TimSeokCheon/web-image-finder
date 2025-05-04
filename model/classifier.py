"""
Image Similarity Classification Module

Contains the ImageSimilarityClassifier class for comparing images based on feature similarity.
"""
import os
import numpy as np
import pickle
import time
from sklearn.neighbors import NearestNeighbors

from utils import config
from model.feature_extractor import extract_features, batch_extract_features

class ImageSimilarityClassifier:
    """
    A similarity-based classifier that uses feature vectors to determine
    if new images are similar to a set of training images.
    """
    def __init__(self, threshold=None):
        """
        Initialize the classifier.
        
        Args:
            threshold (float): Similarity threshold (0-1). Higher means more strict.
                              If None, uses the value from config.
        """
        self.features = []
        self.image_paths = []
        self.nn_model = None
        self.threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD
        
    def train(self, image_folder, recursive=None, batch_size=None):
        """
        Learn from all images in a folder, including subfolders (albums).
        
        Args:
            image_folder (str): Path to folder containing training images.
            recursive (bool): Whether to include images from subfolders (albums)
                             If None, uses the value from config.
            batch_size (int): Batch size for feature extraction. If None, uses the value from config.
        """
        from model.feature_extractor import get_all_images_from_folder
        
        # Use configured values if not specified
        recursive = recursive if recursive is not None else config.USE_RECURSIVE_SCAN
        batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
        
        print(f"Training on images in {image_folder}...")
            
        # Get all image files including from nested albums if recursive=True
        image_paths = get_all_images_from_folder(image_folder, recursive=recursive)
            
        if not image_paths:
            raise ValueError(f"No images found in {image_folder}")
        
        start_time = time.time()
        print(f"Starting feature extraction with batch size: {batch_size} - this can take some time...")
        
        # Extract features in batches - much faster than processing images one by one
        features_list, successful_paths = batch_extract_features(image_paths, batch_size=batch_size)
        self.image_paths = successful_paths
        self.features = np.array(features_list)
        
        # Train nearest neighbors model
        print("Training nearest neighbors model...")
        self.nn_model = NearestNeighbors(
            n_neighbors=min(config.NUM_NEIGHBORS, len(self.features)), 
            metric=config.METRIC
        )
        self.nn_model.fit(self.features)
        
        elapsed_time = time.time() - start_time
        images_per_second = len(successful_paths) / elapsed_time
        print(f"Training completed in {elapsed_time:.2f} seconds ({images_per_second:.2f} images/second)")
        print(f"Successfully processed {len(successful_paths)} out of {len(image_paths)} images")
    
    def _extract_features_for_comparison(self, image_path):
        """Internal method to extract features for similarity check"""
        return extract_features(image_path)
    
    def _check_similarity_with_features(self, features):
        """Internal method to check similarity using pre-extracted features"""
        features_np = features.cpu().numpy().reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(features_np)
        similarity_score = 1 - distances[0][0]
        return similarity_score >= self.threshold, similarity_score, indices
    
    def check_similarity(self, image_path, return_similar=False):
        """
        Check if the image is similar to any in the training set.
        
        Args:
            image_path (str): Path to the test image.
            return_similar (bool): Whether to return the most similar images.
            
        Returns:
            bool: True if similar, False otherwise
            list (optional): List of similar image paths if return_similar is True
        """
        # Extract features for the test image
        features = self._extract_features_for_comparison(image_path)
        
        # Check similarity
        is_similar, similarity_score, indices = self._check_similarity_with_features(features)
        
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Similarity score: {similarity_score:.4f} (Threshold: {self.threshold})")
        print(f"Result: {'Similar' if is_similar else 'Not similar'}")
        
        if return_similar:
            similar_images = [self.image_paths[idx] for idx in indices[0]]
            return is_similar, similar_images
        return is_similar
    
    def save_model(self, path=None):
        """
        Save the trained model to a file.
        
        Args:
            path (str): Path to save the model. If None, uses config values.
        """
        if self.nn_model is None:
            raise ValueError("Model not trained yet")
        
        if path is None:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
        
        save_data = {
            'features': self.features,
            'image_paths': self.image_paths,
            'threshold': self.threshold
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """
        Load a previously trained model from a file.
        
        Args:
            path (str): Path to load the model from. If None, uses config values.
        """
        if path is None:
            path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
            
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.features = save_data['features']
        self.image_paths = save_data['image_paths']
        
        # Always use the current threshold from config.py, ignoring the saved value
        self.threshold = config.SIMILARITY_THRESHOLD
        print(f"Using current threshold from config: {self.threshold}")
        
        self.nn_model = NearestNeighbors(
            n_neighbors=min(config.NUM_NEIGHBORS, len(self.features)), 
            metric=config.METRIC
        )
        self.nn_model.fit(self.features)
        print(f"Model loaded from {path} with {len(self.features)} images")