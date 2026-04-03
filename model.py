import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        
        # Use EfficientNet-B0 as base model (better performance/speed trade-off)
        self.feature_extractor = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers
        for param in list(self.feature_extractor.parameters())[:-30]:
            param.requires_grad = False
            
        # Remove classifier
        num_features = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier = nn.Identity()
        
        # Enhanced classifier with attention and regularization
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features with EfficientNet
        features = self.feature_extractor(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Final classification
        output = self.classifier(attended_features)
        return output.squeeze() 