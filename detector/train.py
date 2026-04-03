import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from utils import AdvancedDeepfakeDetector, process_image, process_video
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_video=False, augment=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_video = is_video
        self.augment = augment
        self.classes = ['real', 'fake']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                logging.warning(f"Directory not found: {class_dir}")
                continue
                
            for filename in os.listdir(class_dir):
                if is_video and filename.endswith(('.mp4', '.avi')):
                    self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))
                elif not is_video and filename.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))
        
        logging.info(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            if self.is_video:
                frames = process_video(path, max_frames=45)
                if self.transform:
                    frames = torch.stack([self.transform(frame) for frame in frames])
                return frames, label
            else:
                image = process_image(path)
                if self.transform:
                    image = self.transform(image)
                return image, label
        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")
            # Return a dummy sample in case of error
            if self.is_video:
                return torch.zeros((1, 45, 3, 224, 224)), label
            else:
                return torch.zeros((1, 3, 224, 224)), label

def get_transforms(is_train=True):
    """Get data transforms for training or validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, aux_outputs = model(inputs)
            
            # Calculate main loss
            loss = criterion(outputs.squeeze(), labels.float())
            
            # Add auxiliary losses
            aux_loss = sum(criterion(aux.squeeze(), labels.float()) for aux in aux_outputs.values())
            loss = loss + 0.1 * aux_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs.squeeze() > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs, aux_outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                
                val_loss += loss.item()
                predictions = (outputs.squeeze() > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        # Log epoch statistics
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_model.pth')
            logging.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info('Early stopping triggered')
                break

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')
        
        # Data transforms
        train_transform = get_transforms(is_train=True)
        val_transform = get_transforms(is_train=False)
        
        # Create datasets
        train_dataset = DeepfakeDataset('data/train', transform=train_transform, augment=True)
        val_dataset = DeepfakeDataset('data/val', transform=val_transform, augment=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Initialize model
        model = AdvancedDeepfakeDetector().to(device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        # Train model
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device=device)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 