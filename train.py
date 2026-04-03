import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from model import DeepfakeDetector
import logging
import gc
import time
from datetime import datetime
import warnings

# Suppress CUDA/cuDNN warnings that aren't critical
warnings.filterwarnings("ignore", category=UserWarning)

# Try importing CUDA components with error handling
try:
    from torch.cuda.amp import GradScaler
    import torch.amp as amp
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    # Define stubs for GradScaler and amp if not available
    class GradScalerStub:
        def __init__(self, *args, **kwargs): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def state_dict(self): return {}
        def load_state_dict(self, state_dict): pass

    GradScaler = GradScalerStub
    
    # Create a stub for amp.autocast if not available
    class AmpStub:
        class autocast:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
    
    amp = AmpStub

# Setup logging
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join('logs', f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return timestamp

# Performance-optimized transforms
train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DeepfakeDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        real_dir = os.path.join(split, 'Real')
        fake_dir = os.path.join(split, 'Fake')
        
        # Load real images
        if os.path.exists(real_dir):
            real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.images.extend(real_images)
            self.labels.extend([0] * len(real_images))
        
        # Load fake images
        if os.path.exists(fake_dir):
            fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.images.extend(fake_images)
            self.labels.extend([1] * len(fake_images))
        
        logging.info(f"{split} dataset: {len(self.images)} images (Real: {len(real_images)}, Fake: {len(fake_images)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            # Return a zero tensor if image loading fails
            return torch.zeros((3, 224, 224)), self.labels[idx]

def find_optimal_workers():
    """Find optimal number of workers based on system resources"""
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
    # For GTX 1650 and i5 10th Gen with 8GB RAM, 2-4 workers is typically optimal
    # Higher values can cause memory pressure
    return min(4, max(2, cpu_count - 2))

def optimize_cuda():
    """Optimize CUDA settings for GTX 1650, or fall back to CPU"""
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available! Training will run on CPU (much slower).")
        logging.warning("To use GPU, check your NVIDIA drivers and PyTorch CUDA installation.")
        return torch.device('cpu')
    
    # Enable cuDNN autotuner for GTX 1650
    torch.backends.cudnn.benchmark = True
    
    # Optimize memory usage
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device('cuda:0')
    
    # Log GPU info
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"CUDA Version: {torch.version.cuda}")
    logging.info(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_val_acc, log_dir, is_best=False):
    """Save model checkpoint with proper error handling"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'scaler_state_dict': scaler.state_dict(),
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # If best model, save separately
        if is_best:
            best_model_path = os.path.join(log_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            logging.info(f"New best model saved: {best_model_path}")
        
        # Remove old checkpoint to save space
        old_checkpoint = os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth')
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)
        
        return True
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")
        return False

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    """Load model checkpoint with error handling"""
    try:
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        
        logging.info(f"Resuming from epoch {start_epoch}, best acc: {best_val_acc:.4f}")
        return start_epoch, best_val_acc
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return 0, 0.0

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, 
                scheduler, num_epochs, device, checkpoint_path, log_dir):
    """Train model with performance optimizations for GPU or CPU"""
    best_val_acc = 0.0
    start_epoch = 0
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Setup mixed precision for CUDA
    use_mixed_precision = device.type == 'cuda'
    if use_mixed_precision:
        # Correct initialization without passing 'cuda' string
        scaler = GradScaler()
    else:
        scaler = None
        logging.info("Mixed precision disabled (using CPU)")
    
    # Load checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if use_mixed_precision and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        logging.info(f"Resuming from epoch {start_epoch}, best acc: {best_val_acc:.4f}")
    
    model = model.to(device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Training loop with performance tracking
    training_start = time.time()
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        # Empty cache before each epoch if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        batch_times = []
        
        for inputs, labels in train_pbar:
            batch_start = time.time()
            
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            
            # Zero gradients efficiently
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with or without mixed precision
            if use_mixed_precision:
                with amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling for mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass for CPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            # Track batch time for performance monitoring
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            if len(batch_times) > 10:
                batch_times.pop(0)
            avg_batch_time = sum(batch_times) / len(batch_times)
            
            # Update progress bar with key metrics
            lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * running_corrects.float() / total:.2f}%',
                'lr': f'{lr:.6f}',
                'bt': f'{avg_batch_time:.3f}s'
            })
            
            # Clean up memory
            del inputs, labels, outputs, loss, preds
        
        # Log epoch training results
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()
                
                # Forward pass with or without mixed precision
                if use_mixed_precision:
                    with amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Calculate accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                # Clean up memory
                del inputs, labels, outputs, loss, preds
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)
        
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        logging.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
        }
        
        # Add scaler state only if using mixed precision
        if use_mixed_precision:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
            
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            # Save best model
            best_model_path = os.path.join(log_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            logging.info(f"New best model saved: {best_model_path}")
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Remove old checkpoint to save space
        old_checkpoint = os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth')
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)
        
        # Early stopping check (optional)
        if train_acc > 0.99 and val_acc > 0.97:
            logging.info("Early stopping criteria met!")
            break
    
    # Final evaluation on test set
    if test_loader:
        model.eval()
        test_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()
                
                if use_mixed_precision:
                    with amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                test_corrects += torch.sum(preds == labels.data)
        
        test_acc = test_corrects.float() / len(test_loader.dataset)
        logging.info(f"Final Test Accuracy: {test_acc:.4f}")
    
    total_time = time.time() - training_start
    logging.info(f"Total training time: {total_time/60:.2f} minutes")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model

def main():
    # Setup logging with timestamp for folder
    timestamp = setup_logging()
    log_dir = os.path.join('logs', timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Try to detect and configure CUDA
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Enable cuDNN autotuner for GTX 1650
            torch.backends.cudnn.benchmark = True
            
            # Optimize memory usage
            torch.backends.cudnn.deterministic = False
            
            # Enable TF32 tensor cores if available
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log GPU info
            logging.info(f"CUDA Version: {torch.version.cuda}")
            logging.info(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Set batch size for GPU (GTX 1650 with 4GB VRAM)
            batch_size = 24
            num_workers = min(4, max(2, os.cpu_count() - 2))
            use_mixed_precision = True
            logging.info("Using mixed precision training (FP16)")
        else:
            device = torch.device('cpu')
            logging.info("CUDA is not available. Using CPU for training (slower).")
            
            # CPU-optimized settings
            batch_size = 16
            num_workers = max(1, os.cpu_count() - 1) 
            use_mixed_precision = False
            logging.info("Mixed precision disabled (using CPU)")
    except Exception as e:
        logging.warning(f"Error detecting CUDA: {e}")
        device = torch.device('cpu')
        batch_size = 16
        num_workers = max(1, os.cpu_count() - 1)
        use_mixed_precision = False
        logging.info("Falling back to CPU due to CUDA detection error")
    
    logging.info(f"Using {num_workers} workers for data loading")
    
    # Initialize model
    model = DeepfakeDetector().to(device)
    
    # Create datasets
    train_dataset = DeepfakeDataset('Train', transform=train_transform)
    val_dataset = DeepfakeDataset('Validate', transform=val_transform)
    test_dataset = DeepfakeDataset('Test', transform=val_transform)
    
    # Create data loaders with performance optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Loss function with class weighting for imbalanced dataset
    pos_weight = torch.tensor([1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Conservative starting learning rate
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - OneCycleLR for faster convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=4e-4,  # Peak learning rate
        epochs=40,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Percentage of training to increase LR
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # Find latest checkpoint - check multiple locations
    latest_checkpoint = None
    
    # First priority: Look for checkpoint_epoch_40.pth (exact epoch 40)
    if os.path.exists('checkpoint_epoch_40.pth'):
        latest_checkpoint = 'checkpoint_epoch_40.pth'
        logging.info(f"Found epoch 40 checkpoint: {latest_checkpoint}")
    else:
        # Look in logs directory for epoch 40 checkpoints
        for log_subdir in os.listdir('logs') if os.path.exists('logs') else []:
            checkpoint_path = os.path.join('logs', log_subdir, 'checkpoint_epoch_40.pth')
            if os.path.exists(checkpoint_path):
                latest_checkpoint = checkpoint_path
                logging.info(f"Found epoch 40 checkpoint in logs: {latest_checkpoint}")
                break
        
        # If no epoch 40 checkpoint found, look for emergency checkpoint
        if latest_checkpoint is None and os.path.exists('checkpoints/emergency_checkpoint.pth'):
            latest_checkpoint = 'checkpoints/emergency_checkpoint.pth'
            logging.info(f"Found emergency checkpoint: {latest_checkpoint}")
        
        # If still no checkpoint, check for any checkpoint files in root
        if latest_checkpoint is None:
            root_checkpoints = [f for f in os.listdir('.') if f.startswith('checkpoint_epoch_')]
            if root_checkpoints:
                latest_checkpoint = max(root_checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
                logging.info(f"Found latest checkpoint: {latest_checkpoint}")
    
    # If no checkpoint found but best_model.pth exists, use that
    if latest_checkpoint is None and os.path.exists('best_model.pth'):
        latest_checkpoint = 'best_model.pth'
        logging.info(f"No checkpoint found, using best model: {latest_checkpoint}")
    
    # Log training configuration
    logging.info(f"Training Configuration:")
    logging.info(f"- Device: {device}")
    logging.info(f"- Model: DeepfakeDetector")
    logging.info(f"- Batch size: {batch_size}")
    logging.info(f"- Max learning rate: {4e-4}")
    logging.info(f"- Workers: {num_workers}")
    logging.info(f"- Total epochs: 40")
    logging.info(f"- Optimizer: AdamW with weight decay")
    logging.info(f"- Scheduler: OneCycleLR")
    if use_mixed_precision and device.type == 'cuda':
        logging.info(f"- Mixed precision: Enabled (FP16)")
    else:
        logging.info(f"- Mixed precision: Disabled")
    
    # Train the model
    train_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader,
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=40,  
        device=device, 
        checkpoint_path=latest_checkpoint,
        log_dir=log_dir
    )
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise