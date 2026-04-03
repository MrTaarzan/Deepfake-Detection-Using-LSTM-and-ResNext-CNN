import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import logging
from detector.utils import AdvancedDeepfakeDetector, process_image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_model(model_path, test_dir):
    # Load model
    model = AdvancedDeepfakeDetector()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test real images
    real_dir = os.path.join(test_dir, 'Real')
    fake_dir = os.path.join(test_dir, 'Fake')
    
    logging.info("\nTesting Real Images:")
    real_correct = 0
    real_total = 0
    for img_name in os.listdir(real_dir)[:10]:  # Test first 10 real images
        img_path = os.path.join(real_dir, img_name)
        try:
            # Process image
            img_tensor = process_image(img_path)
            img_tensor = img_tensor.to(device)
            
            # Get prediction
            with torch.no_grad():
                output, confidence = model(img_tensor)
                prediction = torch.sigmoid(output).item()
                
            # Log results
            is_fake = prediction > 0.5
            real_total += 1
            if not is_fake:
                real_correct += 1
            logging.info(f"{img_name}: Prediction={prediction:.4f}, Confidence={confidence.item():.4f}, Is Fake={is_fake}")
            
        except Exception as e:
            logging.error(f"Error processing {img_name}: {str(e)}")
    
    # Test fake images
    logging.info("\nTesting Fake Images:")
    fake_correct = 0
    fake_total = 0
    for img_name in os.listdir(fake_dir)[:10]:  # Test first 10 fake images
        img_path = os.path.join(fake_dir, img_name)
        try:
            # Process image
            img_tensor = process_image(img_path)
            img_tensor = img_tensor.to(device)
            
            # Get prediction
            with torch.no_grad():
                output, confidence = model(img_tensor)
                prediction = torch.sigmoid(output).item()
                
            # Log results
            is_fake = prediction > 0.5
            fake_total += 1
            if is_fake:
                fake_correct += 1
            logging.info(f"{img_name}: Prediction={prediction:.4f}, Confidence={confidence.item():.4f}, Is Fake={is_fake}")
            
        except Exception as e:
            logging.error(f"Error processing {img_name}: {str(e)}")
    
    # Print summary
    logging.info("\nTest Results Summary:")
    logging.info(f"Real Images: {real_correct}/{real_total} correct ({real_correct/real_total*100:.2f}%)")
    logging.info(f"Fake Images: {fake_correct}/{fake_total} correct ({fake_correct/fake_total*100:.2f}%)")
    logging.info(f"Overall Accuracy: {(real_correct + fake_correct)/(real_total + fake_total)*100:.2f}%")

if __name__ == '__main__':
    test_model('models/best_model.pth', 'Validate') 