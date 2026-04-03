import os
import requests
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'download_samples_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def download_file(url, filename):
    """Download a file from URL to filename."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Successfully downloaded {filename}")
            return True
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
    return False

def main():
    # Create directories
    os.makedirs('sample_data/real', exist_ok=True)
    os.makedirs('sample_data/fake', exist_ok=True)
    
    # Sample videos (replace these with actual URLs)
    samples = {
        'real': [
            'https://example.com/real1.mp4',  # Replace with actual URLs
            'https://example.com/real2.mp4',
        ],
        'fake': [
            'https://example.com/fake1.mp4',
            'https://example.com/fake2.mp4',
        ]
    }
    
    # Download samples
    for label, urls in samples.items():
        for i, url in enumerate(urls):
            filename = f'sample_data/{label}/sample_{i+1}.mp4'
            download_file(url, filename)
    
    logging.info("Sample data download completed!")

if __name__ == "__main__":
    main() 