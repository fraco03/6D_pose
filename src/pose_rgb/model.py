import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torchvision

class ResNetQuaternion(nn.Module):
    """
    ResNet-based 6D Pose Estimation Network
    
    Predicts:
    - Rotation as quaternion (4 values: w, x, y, z)
    - Translation as 3D vector (tx, ty, tz)
    
    Uses camera intrinsics (cam_K) to improve translation estimation.
    """
    
    def __init__(self, freeze_backbone: bool = True):
        """
        Args:
            freeze_backbone: If True, freeze ResNet backbone weights
        """
        super(ResNetQuaternion, self).__init__()
        
        # ResNet backbone
        resnet = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = resnet.fc.in_features  # 2048 for ResNet50
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Camera intrinsics processing
        # Process cam_K: extract fx, fy, cx, cy (4 values)
        self.camera_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Fusion layer: combine image features + camera features
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        fused_dim = 512
        
        # Rotation head (quaternion)
        self.fc_rot = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
        
        # Translation head
        self.fc_trans = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )

    def forward(self, x, cam_K):
        """
        Forward pass
        
        Args:
            x: (B, 3, 224, 224) RGB image crops
            cam_K: (B, 3, 3) camera intrinsic matrices
        
        Returns:
            rot: (B, 4) normalized quaternion [w, x, y, z]
            trans: (B, 3) translation vector [tx, ty, tz]
        """
        # Extract image features
        img_feat = self.features(x)  # (B, 2048, 1, 1)
        img_feat = torch.flatten(img_feat, 1)  # (B, 2048)
        
        # Extract key camera parameters
        # cam_K = [[fx,  0, cx],
        #          [ 0, fy, cy],
        #          [ 0,  0,  1]]
        fx = cam_K[:, 0, 0]  # (B,)
        fy = cam_K[:, 1, 1]  # (B,)
        cx = cam_K[:, 0, 2]  # (B,)
        cy = cam_K[:, 1, 2]  # (B,)
        
        cam_params = torch.stack([fx, fy, cx, cy], dim=1)  # (B, 4)
        cam_feat = self.camera_fc(cam_params)  # (B, 128)
        
        # Fuse image and camera features
        combined = torch.cat([img_feat, cam_feat], dim=1)  # (B, 2176)
        fused = self.fusion(combined)  # (B, 512)
        
        # Predict rotation (quaternion)
        rot = self.fc_rot(fused)  # (B, 4)
        rot = F.normalize(rot, p=2, dim=1)  # Normalize to unit quaternion
        
        # Predict translation
        trans = self.fc_trans(fused)  # (B, 3)
        
        return rot, trans