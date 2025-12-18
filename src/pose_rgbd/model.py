import torch
import torch.nn as nn
import torchvision.models as models

class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = True):
        """
        Args:
            num_classes (int): Number of output features (4 for quaternion rotation).
            pretrained (bool): Whether to use a pretrained ResNet50 backbone.
            freeze_backbone (bool): Whether to freeze the backbone during initial training.
        """
        super(RotationPredictionModel, self).__init__()
        
        # Load ResNet50 backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Replace the fully connected layer with a custom head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the classification layer
        
        # Small CNN for depth processing
        self.depth_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Combine ResNet features and depth features
        self.fc = nn.Sequential(
            nn.Linear(num_features + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Output 4 values for quaternion
        )
        
        # Freeze the backbone if specified
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, rgb, depth):
        # Extract features from ResNet backbone
        rgb_features = self.backbone(rgb)  # Shape: (B, num_features)
        
        # Extract features from depth map
        depth_features = self.depth_net(depth)  # Shape: (B, 32)
        
        # Concatenate features
        combined_features = torch.cat((rgb_features, depth_features), dim=1)
        
        # Predict rotation
        rotation = self.fc(combined_features)
        
        # Normalize quaternion to ensure valid rotation
        rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)
        return rotation