import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from utils import AdvancedDeepfakeDetector, process_image, process_video
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'testing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class DeepfakeTester:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logging.info(f"Using device: {self.device}")
        
        # Load model
        self.model = AdvancedDeepfakeDetector().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict_image(self, image_path):
        """Predict if an image is real or fake."""
        try:
            # Process image
            image = process_image(image_path)
            image = image.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output, aux_outputs = self.model(image)
                prediction = output.squeeze().item()
                
            # Get auxiliary predictions
            aux_predictions = {
                name: aux.squeeze().item() 
                for name, aux in aux_outputs.items()
            }
            
            # Combine predictions
            final_prediction = 0.7 * prediction + 0.1 * sum(aux_predictions.values())
            
            return {
                'prediction': final_prediction,
                'auxiliary_predictions': aux_predictions,
                'is_fake': final_prediction > 0.5
            }
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return None
            
    def predict_video(self, video_path):
        """Predict if a video contains deepfakes."""
        try:
            # Process video
            frames = process_video(video_path, max_frames=45)
            frames = frames.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output, aux_outputs = self.model(frames)
                prediction = output.squeeze().item()
                
            # Get auxiliary predictions
            aux_predictions = {
                name: aux.squeeze().item() 
                for name, aux in aux_outputs.items()
            }
            
            # Combine predictions
            final_prediction = 0.7 * prediction + 0.1 * sum(aux_predictions.values())
            
            return {
                'prediction': final_prediction,
                'auxiliary_predictions': aux_predictions,
                'is_fake': final_prediction > 0.5
            }
            
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {str(e)}")
            return None
            
    def test_directory(self, directory_path, is_video=False):
        """Test all images or videos in a directory."""
        results = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if is_video and filename.endswith(('.mp4', '.avi')):
                result = self.predict_video(file_path)
            elif not is_video and filename.endswith(('.jpg', '.jpeg', '.png')):
                result = self.predict_image(file_path)
            else:
                continue
                
            if result:
                results.append({
                    'filename': filename,
                    **result
                })
                
        return results

def main():
    try:
        # Initialize tester
        tester = DeepfakeTester('best_model.pth')
        
        # Test single image
        image_path = "path/to/test/image.jpg"  # Update this path
        if os.path.exists(image_path):
            result = tester.predict_image(image_path)
            if result:
                logging.info(f"Image {image_path}:")
                logging.info(f"Prediction: {result['prediction']:.4f}")
                logging.info(f"Is Fake: {result['is_fake']}")
                logging.info("Auxiliary Predictions:")
                for name, pred in result['auxiliary_predictions'].items():
                    logging.info(f"  {name}: {pred:.4f}")
        
        # Test single video
        video_path = "path/to/test/video.mp4"  # Update this path
        if os.path.exists(video_path):
            result = tester.predict_video(video_path)
            if result:
                logging.info(f"Video {video_path}:")
                logging.info(f"Prediction: {result['prediction']:.4f}")
                logging.info(f"Is Fake: {result['is_fake']}")
                logging.info("Auxiliary Predictions:")
                for name, pred in result['auxiliary_predictions'].items():
                    logging.info(f"  {name}: {pred:.4f}")
        
        # Test directory of images
        image_dir = "path/to/test/images"  # Update this path
        if os.path.exists(image_dir):
            results = tester.test_directory(image_dir, is_video=False)
            logging.info(f"\nTesting {len(results)} images:")
            for result in results:
                logging.info(f"{result['filename']}: {result['prediction']:.4f} (Fake: {result['is_fake']})")
        
        # Test directory of videos
        video_dir = "path/to/test/videos"  # Update this path
        if os.path.exists(video_dir):
            results = tester.test_directory(video_dir, is_video=True)
            logging.info(f"\nTesting {len(results)} videos:")
            for result in results:
                logging.info(f"{result['filename']}: {result['prediction']:.4f} (Fake: {result['is_fake']})")
                
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
