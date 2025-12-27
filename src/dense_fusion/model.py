import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FusionPoseModel(nn.Module):
    def __init__(self, num_points=1024):
        super(FusionPoseModel, self).__init__()
        
        # BRANCH 1: Geometry (PointNet)
        # Input: (Batch, 3, N)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # BRANCH 2: Color (CNN Encoder - ResNet18)
        # Input: (Batch, 3, 224, 224) -> Feature Vector
        # We use pre-trained ResNet18 (lightweight and fast)
        resnet = models.resnet18(pretrained=True)   #TODO: change to weights=models.ResNet18_Weights.DEFAULT

        # Remove last fully connected layer
        # Output: (Batch, 512, 1, 1)
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Note: ResNet18 outputs 512 canali
        
        # --- FUSION ---
        # Feature PointNet (1024) + Feature RGB (512) = 1536
        self.fc1 = nn.Linear(1024 + 512, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        
        # --- OUTPUT HEADS ---
        # Rotation (Quaternion)
        self.rot_head = nn.Linear(256, 4)
        # Translation (Residual)
        self.trans_head = nn.Linear(256, 3)

    def forward(self, points, images):
        """
        points: (Batch, 3, N)
        images: (Batch, 3, H, W)
        """
        # 1. PointNet Branch
        x = F.relu(self.bn1(self.conv1(points)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Global Max Pooling -> (Batch, 1024)
        feat_geo = torch.max(x, 2, keepdim=False)[0] 
        
        # 2. RGB Branch
        # ResNet output: (Batch, 512, 1, 1)
        feat_rgb = self.rgb_encoder(images)
        # Appiattiamo: (Batch, 512)
        feat_rgb = feat_rgb.view(feat_rgb.size(0), -1)
        
        # 3. Fusion (Concat)
        # (Batch, 1024+512) -> (Batch, 1536)
        global_feat = torch.cat([feat_geo, feat_rgb], dim=1)
        
        # 4. Regressor MLP
        x = F.relu(self.bn_fc1(self.fc1(global_feat)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        
        # 5. Predictions
        pred_q = self.rot_head(x)
        pred_q = F.normalize(pred_q, p=2, dim=1) # Unit quaternion
        
        pred_t_res = self.trans_head(x)
        
        return pred_q, pred_t_res