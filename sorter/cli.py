#!/usr/bin/env python3
"""
Command-line interface for the Image Sorter


This script is a command-line interface (CLI) tool for sorting images based on their similarity. Here's how you can use it:

1. Run the Script
Open a terminal in Visual Studio Code or your system terminal.
Navigate to the directory containing the cli.py file.
Run the script using Python:
2. Available Commands
The script supports two commands: sort and levels.

Command: sort
This command sorts images into two categories: similar and non-similar.

Usage:
Required Argument:
source_dir: The directory containing the images to sort.
Optional Arguments:

--output: Base directory for the sorted output (default: _data/sorted_images).
--threshold or -t: Similarity threshold (0-1). Images with similarity above this value are considered similar (default: from config.SIMILARITY_THRESHOLD).
--name or -n: Name for the sorted folders (default: basename of source_dir).
--model_path: Path to the similarity model file (default: from config.MODEL_DIR and config.MODEL_NAME).
--recursive or -r: Process images in nested folders.
--move: Move images instead of copying them.

Command: levels
This command organizes images into multiple similarity levels.

Usage:

Required Argument:

source_dir: The directory containing the images to sort.
Optional Arguments:

--output: Base directory for the sorted output (default: _data/sorted_images/levels).
--levels: List of similarity thresholds to define levels (default: [0.8, 0.6, 0.4, 0.2]).
--model_path: Path to the similarity model file (default: from config.MODEL_DIR and config.MODEL_NAME).
--recursive or -r: Process images in nested folders.
--move: Move images instead of copying them.
"""

import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sorter_cli')

# Make sure imports work regardless of how the script is run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sorter import ImageSorter
from model.classifier import ImageSimilarityClassifier
from utils import config

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Sort images based on similarity")
    
    subparsers = parser.add_subparsers(dest='command', help='Sorting command to run')
    
    # Basic sort command
    sort_parser = subparsers.add_parser('sort', help='Sort images into similar/non-similar')
    sort_parser.add_argument("source_dir", help="Directory with images to sort")
    sort_parser.add_argument("--output", default=config.OUTPUT_DIR, 
                        help=f"Base directory for output (default: {config.OUTPUT_DIR})")
    sort_parser.add_argument("--threshold", "-t", type=float, default=config.SIMILARITY_THRESHOLD, 
                        help=f"Similarity threshold 0-1 (default: {config.SIMILARITY_THRESHOLD})")
    sort_parser.add_argument("--name", "-n", help="Name for the sorted folders (default: basename of source_dir)")
    sort_parser.add_argument("--model_path", default=os.path.join(config.MODEL_DIR, config.MODEL_NAME), 
                        help=f"Path to similarity model file (default: {os.path.join(config.MODEL_DIR, config.MODEL_NAME)})")
    sort_parser.add_argument("--recursive", "-r", action="store_true", default=False,
                        help="Process images in nested folders")
    sort_parser.add_argument("--move", action="store_true", default=False,
                        help="Move images instead of copying them")
    
    # Advanced sorting by similarity levels
    levels_parser = subparsers.add_parser('levels', help='Sort images into multiple similarity levels')
    levels_parser.add_argument("source_dir", help="Directory with images to sort")
    levels_parser.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, "levels"),
                        help=f"Base directory for output (default: {os.path.join(config.OUTPUT_DIR, 'levels')})")
    levels_parser.add_argument("--levels", type=float, nargs='+', default=config.DEFAULT_SORT_LEVELS,
                        help=f"Similarity threshold levels (default: {config.DEFAULT_SORT_LEVELS})")
    levels_parser.add_argument("--model_path", default=os.path.join(config.MODEL_DIR, config.MODEL_NAME), 
                        help=f"Path to similarity model file (default: {os.path.join(config.MODEL_DIR, config.MODEL_NAME)})")
    levels_parser.add_argument("--recursive", "-r", action="store_true", default=False,
                        help="Process images in nested folders")
    levels_parser.add_argument("--move", action="store_true", default=not config.SORT_COPY_BY_DEFAULT,
                        help=f"Move images instead of copying them (default: {not config.SORT_COPY_BY_DEFAULT})")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    
    if not args.command:
        logger.error("No command specified. Use 'sort' or 'levels'")
        return 1
    
    # Load the classifier
    model_path = args.model_path
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        logger.error("Please train a model first using the collect_images.py script")
        return 1
    
    logger.info(f"Loading model from {model_path}...")
    
    # Initialize classifier with threshold
    threshold = getattr(args, 'threshold', config.SIMILARITY_THRESHOLD)
    classifier = ImageSimilarityClassifier(threshold=threshold)
    classifier.load_model(model_path)
    
    # Create sorter
    sorter = ImageSorter(classifier)
    
    # Determine whether to copy or move
    copy = not getattr(args, 'move', False)
    move_str = "Copying" if copy else "Moving"
    
    if args.command == 'sort':
        logger.info(f"{move_str} and sorting images from {args.source_dir}")
        
        match_count, non_match_count, error_count = sorter.sort_images(
            args.source_dir, 
            args.output,
            args.name,
            copy=copy,
            recursive=args.recursive
        )
        
        logger.info(f"Sort complete! Found {match_count} similar images and {non_match_count} non-similar images")
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during processing")
    
    elif args.command == 'levels':
        logger.info(f"{move_str} and organizing images by similarity levels: {args.levels}")
        
        # Convert custom output structure with category name
        category_name = os.path.basename(os.path.normpath(args.source_dir))
        output_dir = os.path.join(args.output, category_name)
        
        results = sorter.organize_by_similarity(
            args.source_dir,
            output_dir,
            similarity_levels=args.levels,
            copy=copy,
            recursive=args.recursive
        )
        
        logger.info(f"Organization complete!")
        total = sum(count for level, count in results.items() if level != "error")
        logger.info(f"Processed {total} images across {len(args.levels)} similarity levels")
        if results.get("error", 0) > 0:
            logger.warning(f"Encountered {results['error']} errors during processing")
    
    logger.info(f"Results saved to: {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())