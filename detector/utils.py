import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import timm
from einops import rearrange, repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from huggingface_hub import hf_hub_download
import urllib.request

def download_models():
    """Download all required models and store them locally."""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Face detection model files
    model_files = {
        "models/res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "models/deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    }
    
    # Download each file if it doesn't exist
    for file_path, url in model_files.items():
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"Successfully downloaded {file_path}")
            except Exception as e:
                print(f"Error downloading {file_path}: {str(e)}")
                raise

# Download models when module is imported
download_models()

# Load the DNN face detection model
def load_face_detector():
    """Load OpenCV's DNN face detection model."""
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt"
    
    # Create the network
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

# Initialize face detector
face_detector = load_face_detector()

class AdvancedDeepfakeDetector(nn.Module):
    def __init__(self):
        super(AdvancedDeepfakeDetector, self).__init__()
        
        # Load pretrained models
        self.resnet = models.resnet50(pretrained=True)
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Remove classification layers
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # Additional analysis branches
        self.frequency_analyzer = FrequencyAnalyzer()
        self.ela_branch = ELABranch()
        self.noise_analyzer = NoiseAnalyzer()
        
        # Freeze pretrained weights
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
            
        # Feature processing with corrected dimensions
        self.feature_processor = nn.Sequential(
            nn.Linear(2048 + 1280 + 256 + 128 + 128, 1024),  # Corrected input dimension
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Multi-head attention
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
                nn.Softmax(dim=1)
            ) for _ in range(4)  # 4 attention heads
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4, 256),  # Concatenated attention heads
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Ensure input is on the same device as the model
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
            
        # Extract features from both backbones
        features1 = self.resnet(x)
        features2 = self.efficientnet(x)
        
        # Extract additional features
        freq_features = self.frequency_analyzer(x)
        ela_features = self.ela_branch(x)
        noise_features = self.noise_analyzer(x)
        
        # Reshape all features to 2D tensors [batch_size, features]
        features1 = features1.view(features1.size(0), -1)  # [batch, 2048]
        features2 = features2.view(features2.size(0), -1)  # [batch, 1280]
        
        # Ensure all feature tensors are 2D
        if freq_features.dim() > 2:
            freq_features = freq_features.view(freq_features.size(0), -1)
        if ela_features.dim() > 2:
            ela_features = ela_features.view(ela_features.size(0), -1)
        if noise_features.dim() > 2:
            noise_features = noise_features.view(noise_features.size(0), -1)
        
        # Ensure consistent feature dimensions using adaptive pooling
        freq_features = F.adaptive_avg_pool1d(freq_features.unsqueeze(1), 256).squeeze(1)  # [batch, 256]
        ela_features = F.adaptive_avg_pool1d(ela_features.unsqueeze(1), 128).squeeze(1)  # [batch, 128]
        noise_features = F.adaptive_avg_pool1d(noise_features.unsqueeze(1), 128).squeeze(1)  # [batch, 128]
        
        # Concatenate all features
        features = torch.cat((
            features1,  # [batch, 2048]
            features2,  # [batch, 1280]
            freq_features,  # [batch, 256]
            ela_features,  # [batch, 128]
            noise_features  # [batch, 128]
        ), dim=1)  # Total: [batch, 3840]
        
        # Process features
        processed_features = self.feature_processor(features)
        
        # Apply multi-head attention
        attention_outputs = []
        for attention_head in self.attention_heads:
            attention_weights = attention_head(processed_features)
            attended_features = processed_features * attention_weights
            attention_outputs.append(attended_features)
        
        # Concatenate attention outputs
        multi_head_features = torch.cat(attention_outputs, dim=1)
        
        # Final classification
        output = self.classifier(multi_head_features)
        
        # Calculate confidence using softmax for better scaling
        confidence = F.softmax(torch.abs(output), dim=0)
        
        return output, confidence


class FrequencyAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dct_analyzer = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]  # Multi-scale analysis
        ])
        self.wavelet_analyzer = WaveletAnalyzer()

    def forward(self, x):
        # Ensure input is 4D [batch, channels, height, width]
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")

        B, C, H, W = x.shape

        # DCT analysis
        dct = torch.fft.rfft2(x)  # [batch, channels, height, freq]
        magnitude = torch.abs(dct)
        phase = torch.angle(dct)

        # Interpolate magnitude and phase back to original spatial dimensions
        magnitude = F.interpolate(magnitude, size=(
            H, W), mode='bilinear', align_corners=False)
        phase = F.interpolate(phase, size=(
            H, W), mode='bilinear', align_corners=False)

        # Multi-scale frequency features
        dct_features = []
        for conv in self.dct_analyzer:
            dct_features.append(conv(magnitude))
            dct_features.append(conv(phase))

        # Wavelet analysis
        wavelet_features = self.wavelet_analyzer(x)

        # Combine all frequency domain features
        combined = torch.cat(dct_features + [wavelet_features], dim=1)
        return F.adaptive_avg_pool2d(combined, 1).flatten(1)


class WaveletAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_freq = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.low_freq = nn.Conv2d(3, 32, kernel_size=5, padding=2)

    def forward(self, x):
        # Ensure input is 4D [batch, channels, height, width]
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")

        # Ensure tensor is contiguous
        x = x.contiguous()

        # Simulate wavelet decomposition
        high = self.high_freq(x)
        low = self.low_freq(x)

        # Combine features
        return torch.cat([high, low], dim=1)


class ELABranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # Simulate Error Level Analysis
        compressed = F.interpolate(
            F.interpolate(x, scale_factor=0.5, mode='bilinear'),
            scale_factor=2.0, mode='bilinear'
        )
        ela = torch.abs(x - compressed)
        return self.conv(ela).flatten(1)


class NoiseAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # Extract noise patterns
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        noise = x - blurred
        return self.conv(noise).flatten(1)


def create_model(device):
    """Create and load the model."""
    try:
        # Create the model
        model = AdvancedDeepfakeDetector().to(device)
        
        # Try to load from local file first
        model_path = os.path.join('models', 'best_model.pth')
        if os.path.exists(model_path):
            print(f"Found {model_path}, attempting to load...")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Successfully loaded model weights from {model_path}")
            else:
                print(f"Warning: {model_path} does not contain model_state_dict")
        else:
            print(f"Warning: {model_path} not found. Using untrained model")
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        # Return a basic model as fallback
        class FallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(256 * 28 * 28, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 1),
                    nn.Sigmoid()
                )
                
                # Initialize weights
                self._initialize_weights()
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                pred = self.features(x)
                # Calculate confidence based on prediction distance from 0.5
                confidence = torch.abs(pred - 0.5) * 2
                return pred, confidence
        
        print("Using fallback model")
        return FallbackModel().to(device)


def detect_and_align_face(image):
    """Detect and align face in image using OpenCV DNN."""
    try:
        print(f"Starting face detection for: {type(image)}")
        
        # Handle different input types
        if isinstance(image, str):
            print(f"Loading image from path: {image}")
            # Load image from path
            img = cv2.imread(image)
            if img is None:
                print("Failed to load image using cv2.imread")
                raise ValueError(f"Could not load image from path: {image}")
        elif isinstance(image, Image.Image):
            print("Converting PIL Image to OpenCV format")
            # Convert PIL Image to OpenCV format
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            print("Using numpy array directly")
            # If already a numpy array, make a copy to ensure we don't modify the original
            img = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Validate image
        if img is None or img.size == 0:
            print("Empty or invalid image")
            raise ValueError("Empty or invalid image")

        print(f"Image shape: {img.shape}")

        # Ensure image has correct number of channels
        if len(img.shape) == 2:  # Grayscale
            print("Converting grayscale to BGR")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            print("Converting RGBA to BGR")
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif len(img.shape) != 3 or img.shape[2] != 3:
            print(f"Invalid image channels: {img.shape}")
            raise ValueError(f"Invalid image channels: {img.shape}")

        # Get image dimensions
        (h, w) = img.shape[:2]
        print(f"Image dimensions: {w}x{h}")

        # Create blob from image
        print("Creating blob for face detection")
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        # Pass blob through network
        print("Running face detection network")
        face_detector.setInput(blob)
        detections = face_detector.forward()

        # Find faces with confidence > 0.3 (lowered threshold)
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Lowered threshold from 0.5 to 0.3
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append((box.astype("int"), confidence))
                print(f"Found face with confidence: {confidence:.2f}")

        if not faces:
            print("No faces detected with confidence > 0.3")
            # If no face detected, try using the whole image
            print("Using whole image as fallback")
            x, y, w, h = 0, 0, w, h
        else:
            # Get the face with highest confidence
            face = max(faces, key=lambda x: x[1])
            (x, y, w, h) = face[0]
            print(f"Using face with highest confidence: {face[1]:.2f}")

            # Add margin around face (increased margin)
            margin = int(0.5 * max(w, h))  # Increased from 0.3 to 0.5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)

        # Extract face region
        print("Extracting face region")
        face = img[y:y+h, x:x+w]

        # Validate extracted region
        if face.size == 0:
            print("Failed to extract valid face region")
            raise ValueError("Failed to extract valid face region")

        print(f"Extracted face shape: {face.shape}")

        # Apply color correction and normalization
        print("Applying color correction and normalization")
        face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        print(f"Final face shape: {face_rgb.shape}")

        return face_rgb

    except Exception as e:
        print(f"Error in detect_and_align_face: {str(e)}")
        # Return the original image if face detection fails
        if isinstance(image, str):
            try:
                img = cv2.imread(image)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                pass
        elif isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError(f"Could not process image: {str(e)}")


def process_image(image_path):
    """Process image file for deepfake detection."""
    try:
        print(f"Processing image: {image_path}")
        
        # Detect and align face
        print("Detecting and aligning face...")
        face = detect_and_align_face(image_path)
        
        if face is None:
            print("Face detection failed - no face found")
            raise ValueError("Failed to detect face in image")

        print(f"Face detected successfully. Shape: {face.shape}")
        
        # Convert numpy array to PIL Image
        print("Converting to PIL Image...")
        face_pil = Image.fromarray(face)
        
        # Enhanced image transformations with adaptive resizing
        print("Applying transformations...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Base size for initial processing
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Apply transformations and ensure correct shape
        face_tensor = transform(face_pil)
        print(f"Transformed tensor shape: {face_tensor.shape}")

        # Verify tensor shape
        if face_tensor.shape[0] != 3:
            print(f"Invalid channel count: {face_tensor.shape[0]}")
            raise ValueError(f"Expected 3 channels, got {face_tensor.shape[0]}")

        # Add batch dimension
        face_tensor = face_tensor.unsqueeze(0)
        print(f"Final tensor shape: {face_tensor.shape}")
        
        return face_tensor

    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        raise ValueError(f"Error processing image: {str(e)}")


def process_video(video_path, max_frames=45, target_fps=30, target_size=(224, 224)):
    """Process video or GIF file for deepfake detection."""
    try:
        # Check if file is a GIF
        if video_path.lower().endswith('.gif'):
            # Open GIF using PIL
            gif = Image.open(video_path)
            frames = []
            frame_count = 0
            
            # Calculate frame sampling interval
            total_frames = getattr(gif, 'n_frames', 1)
            sampling_interval = max(1, total_frames // max_frames)
            
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            frame_idx = 0
            while frame_count < max_frames:
                try:
                    # Extract frame from GIF
                    gif.seek(frame_idx)
                    frame = gif.convert('RGB')
                    
                    if frame_idx % sampling_interval == 0:
                        # Detect and align face
                        face = detect_and_align_face(frame)
                        
                        # Convert numpy array to PIL Image
                        face_pil = Image.fromarray(face)
                        
                        # Apply transformations
                        face_tensor = transform(face_pil)
                        
                        if face_tensor.shape[0] != 3:
                            print(f"Skipping frame {frame_count}: incorrect channel count {face_tensor.shape[0]}")
                            continue
                        
                        frames.append(face_tensor)
                        frame_count += 1
                    
                    frame_idx += 1
                    
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error processing GIF frame {frame_count}: {str(e)}")
                    continue
                    
        else:
            # Process video file using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate frame sampling interval based on FPS
            fps_ratio = original_fps / target_fps
            sampling_interval = max(1, int(fps_ratio))
            
            frames = []
            frame_count = 0
            frame_idx = 0
            
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sampling_interval == 0:
                    try:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect and align face
                        face = detect_and_align_face(frame_rgb)
                        
                        # Convert numpy array to PIL Image
                        face_pil = Image.fromarray(face)
                        
                        # Apply transformations
                        face_tensor = transform(face_pil)
                        
                        if face_tensor.shape[0] != 3:
                            print(f"Skipping frame {frame_count}: incorrect channel count {face_tensor.shape[0]}")
                            continue
                        
                        frames.append(face_tensor)
                        frame_count += 1
                        
                    except Exception as e:
                        print(f"Error processing video frame {frame_count}: {str(e)}")
                        continue
                
                frame_idx += 1
            
            cap.release()
        
        if not frames:
            raise ValueError("No valid frames could be extracted from the video/GIF")
        
        # Stack frames
        frames_tensor = torch.stack(frames)
        
        # Add batch dimension
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor
        
    except Exception as e:
        raise ValueError(f"Error processing video/GIF: {str(e)}")

