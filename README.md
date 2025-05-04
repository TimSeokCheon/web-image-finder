# Image Collecting and Similarity Project

A comprehensive toolkit for scraping, analyzing, and sorting images based on visual similarity.

## Overview

This project provides tools to:

1. **Scrape images** from web sources based on search terms
2. **Train similarity models** using reference images 
3. **Sort images** based on their visual similarity to reference images

The system uses computer vision and machine learning techniques to analyze image content and determine visual similarity, rather than relying on metadata.

## Project Structure

```
image-collecting-test/
├── _data/                 # Data storage (not tracked in git)
│   ├── _image_corpus/     # Reference images for model training
│   ├── scraped_images/    # Images downloaded by the scraper
│   └── sorted_images/     # Images sorted by similarity
├── _models/               # Saved machine learning models
├── model/                 # Image similarity model implementation
├── scraper/               # Web scraping functionality
├── sorter/                # Image sorting functionality
└── utils/                 # Shared utilities
```

## Setup

### Prerequisites

- Python 3.8+ 
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-collecting-test.git
   cd image-collecting-test
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Scraping Images

The scraper module downloads images from the web based on search terms.

**Command line usage:**
```bash
python -m scraper.cli "search term" [options]
```

**Options:**
- `--output`, `-o`: Output directory (default: `_data/scraped_images`)
- `--num`, `-n`: Number of images to scrape (default: 30)
- `--method`, `-m`: Scraping method to use: `auto`, `selenium`, or `standard` (default: `auto`)

**Example:**
```bash
# Download 20 cat images
python -m scraper.cli "cute cats" --num 20

# Use Selenium for scraping (may require ChromeDriver)
python -m scraper.cli "landscape photography" --method selenium
```

### 2. Training a Similarity Model

Before sorting images, you need to train a model using reference images.

**Command line usage:**
```bash
python -m model.cli train [options]
```

**Options:**
- `--sample_dir`, `-s`: Directory with reference images (default: `_data/_image_corpus`)
- `--model_path`, `-m`: Path to save the model (default: `_models/similarity_model.pkl`)
- `--threshold`, `-t`: Similarity threshold (0-1) (default: 0.7)
- `--recursive`, `-r`: Process nested folders (default: True)

**Example:**
```bash
# Train model using images in a specific folder
python -m model.cli train --sample_dir my_reference_images/ --threshold 0.8
```

### 3. Testing Images Against the Model

Test individual images or directories against the trained model:

**Command line usage:**
```bash
python -m model.cli test [options]
```

**Options:**
- `--test_image`, `-i`: Path to a single image to test
- `--test_dir`, `-d`: Directory with images to test
- `--model_path`, `-m`: Path to the model (default: `_models/similarity_model.pkl`)
- `--threshold`, `-t`: Similarity threshold (0-1) (default: 0.7)
- `--show_similar`, `-s`: Show similar images from the training set
- `--recursive`, `-r`: Process nested folders (when using --test_dir)

**Example:**
```bash
# Test a single image
python -m model.cli test --test_image path/to/image.jpg --show_similar

# Test all images in a directory
python -m model.cli test --test_dir path/to/images/ --threshold 0.75
```

### 4. Getting Model Information

Display information about a trained model:

```bash
python -m model.cli info [--model_path MODEL_PATH]
```

### 5. Sorting Images

Sort images into "similar" and "non-similar" categories:

**Command line usage:**
```bash
python -m sorter.cli sort SOURCE_DIR [options]
```

**Options:**
- `--output`: Base directory for output (default: `_data/sorted_images`)
- `--threshold`, `-t`: Similarity threshold (0-1) (default: 0.7)
- `--name`, `-n`: Name for the sorted folders (default: basename of source_dir)
- `--model_path`: Path to model file (default: `_models/similarity_model.pkl`)
- `--recursive`, `-r`: Process images in nested folders
- `--move`: Move images instead of copying them

**Example:**
```bash
# Sort scraped images
python -m sorter.cli sort _data/scraped_images/ --name my_sort
```

### 6. Organizing Images by Similarity Level

Organize images into multiple categories based on similarity levels:

**Command line usage:**
```bash
python -m sorter.cli levels SOURCE_DIR [options]
```

**Options:**
- `--output`: Base directory for output (default: `_data/sorted_images/levels`)
- `--levels`: Similarity threshold levels (default: 0.8 0.6 0.4 0.2)
- `--model_path`: Path to model file (default: `_models/similarity_model.pkl`)
- `--recursive`, `-r`: Process images in nested folders
- `--move`: Move images instead of copying them

**Example:**
```bash
# Sort images into four similarity levels
python -m sorter.cli levels _data/scraped_images/ --levels 0.9 0.7 0.5 0.3
```

## Practical Workflows

### Basic Workflow

1. **Collect reference images:**
   - Place reference images in `_data/_image_corpus`

2. **Train a model:**
   ```bash
   python -m model.cli train
   ```

3. **Scrape images related to your interest:**
   ```bash
   python -m scraper.cli "your search term"
   ```

4. **Sort the scraped images:**
   ```bash
   python -m sorter.cli sort _data/scraped_images/
   ```

5. **Review the results:**
   - Similar images will be in `_data/sorted_images/{search_term}/matches/`
   - Non-similar images will be in `_data/sorted_images/{search_term}/non_matches/`

### Advanced Workflow

For more fine-grained sorting:

1. **Sort with multiple similarity levels:**
   ```bash
   python -m sorter.cli levels _data/scraped_images/ --levels 0.9 0.7 0.5 0.3
   ```

2. **Test individual images:**
   ```bash
   python -m model.cli test --test_image path/to/image.jpg --show_similar
   ```

## Configuration

Core settings are defined in `utils/config.py`, including:

- Directory paths
- Similarity threshold
- Image parameters
- Scraper settings

Edit this file to customize default behaviors.

## Troubleshooting

- **No images scraped:** Try using the `selenium` method with `--method selenium`
- **Too many false positives:** Increase similarity threshold (e.g., `--threshold 0.8`)
- **Too many false negatives:** Decrease similarity threshold (e.g., `--threshold 0.6`)
- **Model training errors:** Ensure your reference images are valid and accessible