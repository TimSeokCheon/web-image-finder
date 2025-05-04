#!/usr/bin/env python3
"""
Image Sorter - Sort and organize images based on similarity
"""
import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('image_sorter')

class ImageSorter:
    """
    Sort images into different directories based on similarity to reference images
    """
    def __init__(self, classifier, threshold=None):
        """
        Initialize image sorter with a classifier
        
        Parameters:
        -----------
        classifier : object
            An image classifier object with check_similarity method
        threshold : float, optional
            Similarity threshold (0.0-1.0). If provided, overrides the classifier's threshold.
        """
        self.classifier = classifier
        
        # Override threshold if provided
        if threshold is not None:
            self.classifier.threshold = threshold
            
        logger.info(f"ImageSorter initialized with threshold: {self.classifier.threshold}")

    def sort_images(self, source_dir, output_base, category_name=None, copy=True, recursive=False):
        """
        Sort images from source directory into matches/non-matches folders
        
        Parameters:
        -----------
        source_dir : str
            Directory containing images to sort
        output_base : str
            Base directory for output (matches/non-matches will be created inside)
        category_name : str, optional
            Name for the output subfolder. If None, uses basename of source_dir
        copy : bool, default=True
            If True, copy images to destination. If False, move them.
        recursive : bool, default=False
            If True, process images in subdirectories
            
        Returns:
        --------
        tuple
            (match_count, non_match_count, error_count)
        """
        # Get name for the sorted folders
        if category_name is None:
            category_name = os.path.basename(os.path.normpath(source_dir))
        
        # Setup output directories
        output_dir = os.path.join(output_base, category_name)
        matches_dir = os.path.join(output_dir, "matches")
        non_matches_dir = os.path.join(output_dir, "non_matches")
        
        # Create directories if they don't exist
        os.makedirs(matches_dir, exist_ok=True)
        os.makedirs(non_matches_dir, exist_ok=True)
        
        logger.info(f"Processing images from: {source_dir}")
        logger.info(f"Matches will be saved to: {matches_dir}")
        logger.info(f"Non-matches will be saved to: {non_matches_dir}")
        
        # Get all images from the source directory
        try:
            from image_project.model.feature_extractor import get_all_images_from_folder
            images = get_all_images_from_folder(source_dir, recursive=recursive)
        except ImportError:
            # Fall back to our own implementation if not available
            images = self._get_image_files(source_dir, recursive)
        
        if not images:
            logger.warning(f"No images found in {source_dir}")
            return 0, 0, 0
            
        logger.info(f"Found {len(images)} images to process")
        
        # Process each image
        match_count = 0
        non_match_count = 0
        error_count = 0
        
        for idx, img_path in enumerate(images):
            try:
                logger.info(f"Processing image {idx+1}/{len(images)}: {os.path.basename(img_path)}")
                is_similar = self.classifier.check_similarity(img_path)
                
                # Copy/move the image to the appropriate folder
                file_operation = shutil.copy2 if copy else shutil.move
                
                if is_similar:
                    dest_path = os.path.join(matches_dir, os.path.basename(img_path))
                    file_operation(img_path, dest_path)
                    match_count += 1
                else:
                    dest_path = os.path.join(non_matches_dir, os.path.basename(img_path))
                    file_operation(img_path, dest_path)
                    non_match_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                error_count += 1
        
        logger.info(f"Processing complete!")
        logger.info(f"Total images processed: {match_count + non_match_count}")
        logger.info(f"Similar matches: {match_count}")
        logger.info(f"Non-matches: {non_match_count}")
        logger.info(f"Errors: {error_count}")
        
        return match_count, non_match_count, error_count
    
    def organize_by_similarity(self, source_dir, output_dir, similarity_levels=None, 
                              copy=True, recursive=False):
        """
        Organize images into different directories based on similarity levels
        
        Parameters:
        -----------
        source_dir : str
            Directory containing images to sort
        output_dir : str
            Base directory for output 
        similarity_levels : list of float, optional
            List of similarity thresholds to use. Default: [0.8, 0.6, 0.4, 0.2]
        copy : bool, default=True
            If True, copy images to destination. If False, move them.
        recursive : bool, default=False
            If True, process images in subdirectories
            
        Returns:
        --------
        dict
            Dictionary with counts for each similarity level
        """
        if similarity_levels is None:
            similarity_levels = [0.8, 0.6, 0.4, 0.2]
            
        # Sort similarity levels in descending order
        similarity_levels = sorted(similarity_levels, reverse=True)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a directory for each similarity level
        level_dirs = {}
        for level in similarity_levels:
            level_name = f"similarity_{int(level * 100)}"
            level_dir = os.path.join(output_dir, level_name)
            os.makedirs(level_dir, exist_ok=True)
            level_dirs[level] = level_dir
            
        # Get all images from the source directory
        try:
            from image_project.model.feature_extractor import get_all_images_from_folder
            images = get_all_images_from_folder(source_dir, recursive=recursive)
        except ImportError:
            # Fall back to our own implementation if not available
            images = self._get_image_files(source_dir, recursive)
            
        if not images:
            logger.warning(f"No images found in {source_dir}")
            return {}
            
        logger.info(f"Found {len(images)} images to process")
        
        # Keep track of counts for each similarity level
        counts = {level: 0 for level in similarity_levels}
        counts["other"] = 0  # For images below the lowest threshold
        error_count = 0
        
        # Save the original threshold to restore later
        original_threshold = self.classifier.threshold
        
        # Create a directory for images below the lowest threshold
        other_dir = os.path.join(output_dir, "similarity_other")
        os.makedirs(other_dir, exist_ok=True)
        
        # Process each image
        for idx, img_path in enumerate(images):
            try:
                logger.info(f"Processing image {idx+1}/{len(images)}: {os.path.basename(img_path)}")
                
                # Get features once to avoid recomputing for each threshold
                features = self.classifier._extract_features_for_comparison(img_path)
                
                # Find the highest similarity level where the image is similar
                placed = False
                for level in similarity_levels:
                    # Temporarily set threshold to the current level
                    self.classifier.threshold = level
                    
                    # Check similarity using the pre-computed features
                    is_similar = self.classifier._check_similarity_with_features(features)
                    
                    if is_similar:
                        dest_path = os.path.join(level_dirs[level], os.path.basename(img_path))
                        if copy:
                            shutil.copy2(img_path, dest_path)
                        else:
                            shutil.move(img_path, dest_path)
                        counts[level] += 1
                        placed = True
                        break
                
                # If not placed in any similarity level, put in "other"
                if not placed:
                    dest_path = os.path.join(other_dir, os.path.basename(img_path))
                    if copy:
                        shutil.copy2(img_path, dest_path)
                    else:
                        shutil.move(img_path, dest_path)
                    counts["other"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                error_count += 1
        
        # Restore original threshold
        self.classifier.threshold = original_threshold
        
        # Log results
        logger.info(f"Processing complete!")
        total = sum(counts.values())
        logger.info(f"Total images processed: {total}")
        for level in similarity_levels:
            logger.info(f"Similarity {int(level * 100)}%: {counts[level]} images")
        logger.info(f"Below threshold: {counts['other']} images")
        if error_count > 0:
            logger.info(f"Errors: {error_count}")
        
        counts["error"] = error_count
        return counts
    
    def _get_image_files(self, directory, recursive=False):
        """
        Get all image files in a directory
        
        Parameters:
        -----------
        directory : str
            Directory to scan
        recursive : bool, default=False
            If True, scan subdirectories
            
        Returns:
        --------
        list
            List of image file paths
        """
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        images = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        images.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, file)) and any(file.lower().endswith(ext) for ext in extensions):
                    images.append(os.path.join(directory, file))
        
        return images