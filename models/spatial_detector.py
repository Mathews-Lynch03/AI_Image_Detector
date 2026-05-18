"""
Spatial-Domain CNN Detector
Uses ResNet with transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models

class SpatialDetector(nn.Module):
    """CNN-based detector using transfer learning"""
    
    def __init__(self, model_name='resnet18', pretrained=True, freeze_layers=10):
        super().__init__()
        
        self.model_name = model_name
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze early layers
        if pretrained and freeze_layers > 0:
            layers = list(self.backbone.children())[:freeze_layers]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable