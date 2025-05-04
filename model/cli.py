#!/usr/bin/env python3
"""
Command-line interface for the Image Similarity Model
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
logger = logging.getLogger('model_cli')

# Make sure imports work regardless of how the script is run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our implementation
from model.classifier import ImageSimilarityClassifier
from model.feature_extractor import get_all_images_from_folder
from utils import config

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Image Similarity Model CLI")
    
    subparsers = parser.add_subparsers(dest='command', help='Model command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new similarity model')
    train_parser.add_argument("--sample_dir", "-s", default=config.DATA_DIR,
                        help=f"Directory with sample/reference images (default: {config.DATA_DIR})")
    train_parser.add_argument("--model_path", "-m", default=os.path.join(config.MODEL_DIR, config.MODEL_NAME),
                        help=f"Path to save the model (default: {os.path.join(config.MODEL_DIR, config.MODEL_NAME)})")
    train_parser.add_argument("--threshold", "-t", type=float, default=config.SIMILARITY_THRESHOLD,
                        help=f"Similarity threshold 0-1 (default: {config.SIMILARITY_THRESHOLD})")
    train_parser.add_argument("--recursive", "-r", action="store_true", default=config.USE_RECURSIVE_SCAN,
                        help="Process images in nested folders")
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test images against the similarity model')
    test_parser.add_argument("--test_image", "-i", help="Path to a single image to test")
    test_parser.add_argument("--test_dir", "-d", help="Directory with images to test (either this or --test_image required)")
    test_parser.add_argument("--model_path", "-m", default=os.path.join(config.MODEL_DIR, config.MODEL_NAME),
                       help=f"Path to the model file (default: {os.path.join(config.MODEL_DIR, config.MODEL_NAME)})")
    test_parser.add_argument("--threshold", "-t", type=float, default=config.SIMILARITY_THRESHOLD,
                       help=f"Similarity threshold 0-1 (default: {config.SIMILARITY_THRESHOLD})")
    test_parser.add_argument("--show_similar", "-s", action="store_true", help="Show similar images from the training set")
    test_parser.add_argument("--recursive", "-r", action="store_true", default=False,
                       help="Process images in nested folders (when using --test_dir)")
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display information about a trained model')
    info_parser.add_argument("--model_path", "-m", default=os.path.join(config.MODEL_DIR, config.MODEL_NAME),
                        help=f"Path to the model file (default: {os.path.join(config.MODEL_DIR, config.MODEL_NAME)})")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    
    if not args.command:
        logger.error("No command specified. Use 'train', 'test', or 'info'")
        return 1
    
    if args.command == 'train':
        return train_model(args)
    elif args.command == 'test':
        return test_model(args)
    elif args.command == 'info':
        return model_info(args)

def train_model(args):
    """Train a new similarity model"""
    logger.info(f"Training similarity model with images from: {args.sample_dir}")
    logger.info(f"Using threshold: {args.threshold}")
    logger.info(f"Recursive search: {args.recursive}")
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Initialize classifier with threshold
    classifier = ImageSimilarityClassifier(threshold=args.threshold)
    
    try:
        # Train the model
        classifier.train(args.sample_dir, recursive=args.recursive)
        
        # Save the model
        classifier.save_model(args.model_path)
        logger.info(f"Model trained and saved to: {args.model_path}")
        return 0
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return 1

def test_model(args):
    """Test images against a trained model"""
    # Check if either test_image or test_dir is provided
    if not args.test_image and not args.test_dir:
        logger.error("Either --test_image or --test_dir must be provided")
        return 1
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found at {args.model_path}")
        logger.error("Please train a model first using the 'train' command")
        return 1
    
    logger.info(f"Loading model from {args.model_path}...")
    classifier = ImageSimilarityClassifier(threshold=args.threshold)
    classifier.load_model(args.model_path)
    
    match_count = 0
    total_count = 0
    
    if args.test_image:
        # Test a single image
        try:
            if args.show_similar:
                is_similar, similar_images = classifier.check_similarity(args.test_image, return_similar=True)
                if is_similar:
                    logger.info("Most similar images from training set:")
                    for img in similar_images:
                        logger.info(f"- {os.path.basename(img)}")
            else:
                is_similar = classifier.check_similarity(args.test_image)
            
            match_count = 1 if is_similar else 0
            total_count = 1
            
        except Exception as e:
            logger.error(f"Error processing {args.test_image}: {e}")
            return 1
    else:
        # Test all images in directory
        try:
            logger.info(f"Testing images in {args.test_dir}")
            images = get_all_images_from_folder(args.test_dir, recursive=args.recursive)
            
            if not images:
                logger.error(f"No images found in {args.test_dir}")
                return 1
            
            logger.info(f"Found {len(images)} images to test")
            total_count = len(images)
            
            for idx, img_path in enumerate(images):
                try:
                    logger.info(f"Testing image {idx+1}/{len(images)}: {os.path.basename(img_path)}")
                    if args.show_similar:
                        is_similar, similar_images = classifier.check_similarity(img_path, return_similar=True)
                        if is_similar and similar_images:
                            logger.info("Most similar reference images:")
                            for img in similar_images[:3]:  # Show only top 3 matches
                                logger.info(f"- {os.path.basename(img)}")
                    else:
                        is_similar = classifier.check_similarity(img_path)
                        
                    if is_similar:
                        match_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error testing directory {args.test_dir}: {e}")
            return 1
    
    # Summary
    logger.info(f"Testing complete! Found {match_count} similar images out of {total_count}")
    logger.info(f"Similarity threshold used: {args.threshold}")
    
    return 0

def model_info(args):
    """Display information about a trained model"""
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found at {args.model_path}")
        return 1
    
    try:
        # Load the model to get information
        classifier = ImageSimilarityClassifier()
        classifier.load_model(args.model_path)
        
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Model size: {os.path.getsize(args.model_path) / (1024*1024):.2f} MB")
        logger.info(f"Number of reference images: {len(classifier.features)}")
        logger.info(f"Current similarity threshold: {classifier.threshold}")
        
        # Display some reference image paths
        if classifier.image_paths:
            logger.info("Sample reference images:")
            for path in classifier.image_paths[:5]:  # Show only first 5
                logger.info(f"- {path}")
            if len(classifier.image_paths) > 5:
                logger.info(f"... and {len(classifier.image_paths) - 5} more")
                
        return 0
    except Exception as e:
        logger.error(f"Error reading model information: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())