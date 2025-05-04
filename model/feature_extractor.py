"""
Image feature extraction module using pre-trained ResNet model
"""
import os
from pathlib import Path
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import concurrent.futures
import numpy as np
from tqdm import tqdm
import time

from utils import config

# Use the device configuration from config
device = config.DEVICE
print(f"Using device: {device}")

# Load the pretrained model based on architecture specified in config
if config.MODEL_ARCHITECTURE == 'resnet18':
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
elif config.MODEL_ARCHITECTURE == 'resnet34':
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
elif config.MODEL_ARCHITECTURE == 'resnet50':
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
else:
    # Default to resnet18 if unknown architecture
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
# Remove the final classification layer to get feature vector
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)  # Move model to GPU/CPU as configured
model.eval()  # Set the model to evaluation mode

# Define a transformation pipeline for the input image
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=config.NORMALIZE_MEAN,
        std=config.NORMALIZE_STD
    )
])

def extract_features(image_path):
    """
    Extracts a feature vector from the given image using the configured model.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        torch.Tensor: Feature vector.
    """
    # Load and preprocess the image
    # Handle the transparency warning by explicitly converting to RGB
    with Image.open(image_path) as img:
        # Convert palette images with transparency to RGBA first
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        # Then convert to RGB (removes alpha channel if present)
        image = img.convert('RGB')
        
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to configured device

    # Extract features
    with torch.no_grad():
        features = model(input_tensor).squeeze().cpu()  # Move back to CPU for storage
    return features

def batch_extract_features(image_paths, batch_size=None):
    """
    Extract features from multiple images in batches.
    
    Args:
        image_paths (list): List of image paths
        batch_size (int): Number of images to process at once. If None, use configured batch size.
        
    Returns:
        tuple: (features_list, successful_paths) - Features and paths of successfully processed images
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    features_list = []
    successful_paths = []
    
    # Process images in batches
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    with tqdm(total=len(image_paths), desc="Extracting features") as pbar:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            batch_paths_success = []
            
            # Load and preprocess each image in the batch
            for img_path in batch_paths:
                try:
                    with Image.open(img_path) as img:
                        # Convert palette images with transparency to RGBA first
                        if img.mode == 'P' and 'transparency' in img.info:
                            img = img.convert('RGBA')
                        # Then convert to RGB (removes alpha channel if present)
                        image = img.convert('RGB')
                        
                        # Transform the image
                        tensor = transform(image)
                        batch_tensors.append(tensor)
                        batch_paths_success.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            if batch_tensors:
                # Stack tensors into a batch
                batch = torch.stack(batch_tensors).to(device)
                
                # Process the batch
                with torch.no_grad():
                    batch_features = model(batch).squeeze()
                    
                # Handle single-item batch case
                if len(batch_tensors) == 1:
                    batch_features = batch_features.unsqueeze(0)
                    
                # Store the features and successful paths
                features_list.extend(batch_features.cpu().numpy())
                successful_paths.extend(batch_paths_success)
            
            # Update progress bar
            pbar.update(len(batch_paths))
    
    return features_list, successful_paths

def get_all_images_from_folder(folder_path, recursive=None):
    """
    Get all image files from a folder, including from nested subfolders if recursive=True.
    
    Args:
        folder_path (str): Path to the folder to search
        recursive (bool): Whether to search in nested subfolders. If None, uses config setting.
        
    Returns:
        list: List of image file paths
    """
    all_images = []
    
    # Use configured recursive setting if not specified
    if recursive is None:
        recursive = config.USE_RECURSIVE_SCAN
    
    # Convert to Path object for easier handling
    folder = Path(folder_path)
    
    # If recursive, use rglob to get all files including in subfolders
    if recursive:
        for ext in config.VALID_EXTENSIONS:
            all_images.extend([str(p) for p in folder.rglob(f'*{ext}')])
            all_images.extend([str(p) for p in folder.rglob(f'*{ext.upper()}')])
    else:
        # Use glob for just the current folder
        for ext in config.VALID_EXTENSIONS:
            all_images.extend([str(p) for p in folder.glob(f'*{ext}')])
            all_images.extend([str(p) for p in folder.glob(f'*{ext.upper()}')])
    
    print(f"Found {len(all_images)} images in {folder_path}")
    return all_images