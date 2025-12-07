import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision

class ResNetQuaternion(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super(ResNetQuaternion, self).__init__()
        
        resnet = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_rot = nn.Linear(resnet.fc.in_features, 4)  # Quaternion output

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        rot = self.fc_rot(x)
        rot = rot / (torch.norm(rot, dim=1, keepdim=True) + 1e-8)  # Normalize quaternion
        return rot
    
    def __repr__(self):
        return super().__repr__()