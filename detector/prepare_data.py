import os
import shutil
import random
from pathlib import Path
import requests
import zipfile
import io
import logging
from datetime import datetime
import kaggle
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_preparation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def download_kaggle_dataset(dataset_name, target_dir):
    """Download dataset from Kaggle."""
    try:
        # Ensure Kaggle API is configured
        if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
            logging.error("Kaggle API not configured. Please set up your Kaggle API credentials.")
            return False
            
        # Download dataset
        kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
        logging.info(f"Successfully downloaded dataset: {dataset_name}")
        return True
    except Exception as e:
        logging.error(f"Error downloading Kaggle dataset: {str(e)}")
        return False

def process_kaggle_dataset(source_dir, base_dirs, train_ratio):
    """Process Kaggle dataset format."""
    # Look for metadata files
    metadata_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    
    if not metadata_files:
        logging.error("No metadata file found in the dataset")
        return
        
    # Read metadata
    metadata = pd.read_csv(os.path.join(source_dir, metadata_files[0]))
    
    # Get video files and labels
    video_files = []
    for _, row in metadata.iterrows():
        video_path = os.path.join(source_dir, row['filename'])
        if os.path.exists(video_path):
            video_files.append((video_path, row['label']))
    
    random.shuffle(video_files)
    split_idx = int(len(video_files) * train_ratio)
    
    # Process training set
    for video_path, label in video_files[:split_idx]:
        label_dir = 'fake' if label == 1 else 'real'
        shutil.copy2(video_path, os.path.join(base_dirs['train'][label_dir], os.path.basename(video_path)))
    
    # Process validation set
    for video_path, label in video_files[split_idx:]:
        label_dir = 'fake' if label == 1 else 'real'
        shutil.copy2(video_path, os.path.join(base_dirs['val'][label_dir], os.path.basename(video_path)))
    
    logging.info(f"Processed {len(video_files)} videos:")
    logging.info(f"Training: {split_idx} files")
    logging.info(f"Validation: {len(video_files) - split_idx} files")

def organize_dataset(source_dir, train_ratio=0.8, dataset_type='kaggle'):
    """
    Organize the dataset into train and validation sets.
    
    Args:
        source_dir (str): Path to the source dataset directory
        train_ratio (float): Ratio of data to use for training
        dataset_type (str): Type of dataset ('uadfv', 'faceforensics', 'celebdf', 'kaggle')
    """
    # Create necessary directories
    base_dirs = {
        'train': {'real': 'data/train/real', 'fake': 'data/train/fake'},
        'val': {'real': 'data/val/real', 'fake': 'data/val/fake'}
    }
    
    for split in base_dirs:
        for label in base_dirs[split]:
            os.makedirs(base_dirs[split][label], exist_ok=True)
    
    # Process dataset based on type
    if dataset_type == 'kaggle':
        process_kaggle_dataset(source_dir, base_dirs, train_ratio)
    elif dataset_type == 'uadfv':
        process_uadfv(source_dir, base_dirs, train_ratio)
    elif dataset_type == 'faceforensics':
        process_faceforensics(source_dir, base_dirs, train_ratio)
    elif dataset_type == 'celebdf':
        process_celebdf(source_dir, base_dirs, train_ratio)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def process_uadfv(source_dir, base_dirs, train_ratio):
    """Process UADFV dataset format."""
    for label in ['real', 'fake']:
        source_path = os.path.join(source_dir, label)
        if not os.path.exists(source_path):
            logging.warning(f"Directory not found: {source_path}")
            continue
            
        video_files = [f for f in os.listdir(source_path) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        random.shuffle(video_files)
        split_idx = int(len(video_files) * train_ratio)
        
        for file in video_files[:split_idx]:
            shutil.copy2(
                os.path.join(source_path, file),
                os.path.join(base_dirs['train'][label], file)
            )
            
        for file in video_files[split_idx:]:
            shutil.copy2(
                os.path.join(source_path, file),
                os.path.join(base_dirs['val'][label], file)
            )
            
        logging.info(f"Processed {label} videos:")
        logging.info(f"Training: {split_idx} files")
        logging.info(f"Validation: {len(video_files) - split_idx} files")

def process_faceforensics(source_dir, base_dirs, train_ratio):
    """Process FaceForensics++ dataset format."""
    # FaceForensics++ has a different structure
    video_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_files.append(os.path.join(root, file))
    
    random.shuffle(video_files)
    split_idx = int(len(video_files) * train_ratio)
    
    for file in video_files[:split_idx]:
        label = 'fake' if 'manipulated' in file else 'real'
        shutil.copy2(file, os.path.join(base_dirs['train'][label], os.path.basename(file)))
        
    for file in video_files[split_idx:]:
        label = 'fake' if 'manipulated' in file else 'real'
        shutil.copy2(file, os.path.join(base_dirs['val'][label], os.path.basename(file)))

def process_celebdf(source_dir, base_dirs, train_ratio):
    """Process Celeb-DF dataset format."""
    # Celeb-DF has a different structure
    video_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_files.append(os.path.join(root, file))
    
    random.shuffle(video_files)
    split_idx = int(len(video_files) * train_ratio)
    
    for file in video_files[:split_idx]:
        label = 'fake' if 'id' in file else 'real'
        shutil.copy2(file, os.path.join(base_dirs['train'][label], os.path.basename(file)))
        
    for file in video_files[split_idx:]:
        label = 'fake' if 'id' in file else 'real'
        shutil.copy2(file, os.path.join(base_dirs['val'][label], os.path.basename(file)))

def main():
    try:
        # Set random seed for reproducibility
        random.seed(42)
        
        print("\nWhich dataset would you like to use?")
        print("1. DFDC Preview Dataset (Kaggle)")
        print("2. Deepfake Detection Dataset (Kaggle)")
        print("3. UADFV")
        print("4. FaceForensics++")
        print("5. Celeb-DF")
        
        choice = input("Enter your choice (1-5): ")
        
        dataset_configs = {
            '1': {
                'name': 'xhlulu/dfdc-preview-dataset',
                'type': 'kaggle',
                'target_dir': 'dfdc_preview'
            },
            '2': {
                'name': 'dagnelies/deepfake-detection-dataset',
                'type': 'kaggle',
                'target_dir': 'deepfake_detection'
            },
            '3': {
                'type': 'uadfv',
                'target_dir': 'uadfv'
            },
            '4': {
                'type': 'faceforensics',
                'target_dir': 'faceforensics'
            },
            '5': {
                'type': 'celebdf',
                'target_dir': 'celebdf'
            }
        }
        
        if choice not in dataset_configs:
            raise ValueError("Invalid choice")
            
        config = dataset_configs[choice]
        
        # Download dataset if it's from Kaggle
        if config['type'] == 'kaggle':
            if not download_kaggle_dataset(config['name'], config['target_dir']):
                raise Exception("Failed to download Kaggle dataset")
            source_dir = config['target_dir']
        else:
            source_dir = config['target_dir']
            if not os.path.exists(source_dir):
                raise ValueError(f"Dataset directory not found: {source_dir}")
        
        # Organize the dataset
        organize_dataset(source_dir, train_ratio=0.8, dataset_type=config['type'])
        logging.info("Dataset organization completed!")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 