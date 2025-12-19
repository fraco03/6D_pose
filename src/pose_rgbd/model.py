import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class RotationPredictionModel(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        """
        RGBD Pose Estimation Model using ResNet-50 + Depth CNN.
        
        Args:
            pretrained (bool): If True, loads ImageNet pretrained weights for ResNet-50.
            freeze_backbone (bool): If True, prevents the ResNet weights from being updated 
                                    during the first phase of training. Recommended to avoid 
                                    destroying pretrained features.
        """
        super(RotationPredictionModel, self).__init__()

        # 1. Load Pretrained ResNet-50 for RGB
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        # 2. RGB Feature Extractor
        # Remove the final classification layer (fc) but keep the Global Average Pooling.
        # This takes the input image and outputs a vector of size 2048.
        self.rgb_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet-50 feature dimension is always 2048 before the final layer
        rgb_feature_dim = resnet.fc.in_features 

        # 3. Depth Feature Extractor
        # Small CNN for depth processing - outputs 64D features
        self.depth_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        depth_feature_dim = 64

        # 4. Freeze RGB Backbone (Optional but Recommended)
        if freeze_backbone:
            for param in self.rgb_features.parameters():
                param.requires_grad = False
            print("🔒 ResNet RGB backbone frozen.")

        # 5. Combined Regression Head
        # Maps RGB (2048) + Depth (64) = 2112 features to a 4D quaternion.
        # Multi-layer architecture for non-linear relationships.
        combined_dim = rgb_feature_dim + depth_feature_dim
        
        self.rot_head = nn.Sequential(
            nn.Flatten(),
            
            # Layer 1: High-level feature extraction
            nn.Linear(combined_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            
            # Layer 2: Refinement
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),

            # Layer 3: More Refinement
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            
            # Layer 4: Final Regression to Quaternion
            nn.Linear(256, 4)
        )

        # Initialize the new layers with proper scaling
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the new layers with proper scaling.
        Uses Kaiming (He) initialization for layers with LeakyReLU.
        """
        # Initialize depth_features (CNN layers)
        for m in self.depth_features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize rot_head (fully connected layers)
        for m in self.rot_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def freeze_backbone(self):
        """Freeze all parameters in the RGB backbone."""
        for param in self.rgb_features.parameters():
            param.requires_grad = False
        print("🔒 ResNet RGB backbone frozen.")

    def unfreeze_backbone(self):
        """Unfreeze all parameters in the RGB backbone."""
        for param in self.rgb_features.parameters():
            param.requires_grad = True
        print("🔓 ResNet RGB backbone unfrozen.")

    def forward(self, rgb, depth):
        """
        Forward pass through the model.
        
        Args:
            rgb: RGB image tensor of shape (B, 3, H, W)
            depth: Depth map tensor of shape (B, 1, H, W)
            
        Returns:
            Normalized quaternion tensor of shape (B, 4)
        """
        # Extract features from RGB (ResNet-50)
        rgb_feat = self.rgb_features(rgb)  # Shape: (B, 2048, 1, 1)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)  # Shape: (B, 2048)
        
        # Extract features from depth (CNN)
        depth_feat = self.depth_features(depth)  # Shape: (B, 64)
        
        # Concatenate RGB and depth features
        combined = torch.cat((rgb_feat, depth_feat), dim=1)  # Shape: (B, 2112)
        
        # Predict quaternion through regression head
        quaternion = self.rot_head(combined)  # Shape: (B, 4)
        
        # Normalize quaternion to ensure valid rotation
        quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
        
        return quaternion