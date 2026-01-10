import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class RGBDRotationModel(nn.Module):
    def __init__(self, pretrained=True):
        super(RGBDRotationModel, self).__init__()
        
        # 1. Load Pretrained ResNet-50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Change the first conv layer to accept 4 channels (RGB + Depth)
        original_conv1 = self.backbone.conv1
        
        # New conv layer with 4 input channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels=4, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Smart weight initialization for the new conv layer
        with torch.no_grad():
            # Load original weights for RGB channels
            self.backbone.conv1.weight[:, :3, :, :] = original_conv1.weight
            
            # For the depth channel, initialize with the mean of RGB weights
            self.backbone.conv1.weight[:, 3, :, :] = torch.mean(original_conv1.weight, dim=1)

        # Remove the final classification layer
        self.feature_dim = self.backbone.fc.in_features # 2048
        self.backbone.fc = nn.Identity() # Rimuoviamo il layer finale

        # 2. Quaternion Regression Head
        self.rot_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Linear(512, 4) # Output raw quaternion
        )

    def forward(self, rgb, depth):
        """
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W) -> Depth 'grezza'
        """
        # Concatenate along channel dimension
        x = torch.cat([rgb, depth], dim=1) # (B, 4, H, W)
        
        # Pass through backbone
        features = self.backbone(x) # Output (B, 2048)
        
        # Regress quaternion
        q = self.rot_head(features)
        
        # L2 Normalize quaternion
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        
        return q