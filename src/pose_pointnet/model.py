import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetPoseModel(nn.Module):
    def __init__(self, num_points=1024):
        super(PointNetPoseModel, self).__init__()
        
        # --- PointNet Encoder (MLP su ogni punto) ---
        # Input: (Batch, 3, Num_Points)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # --- Global Feature Processing ---
        # Input: 1024 (Global Feature Vector)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
        
        # --- Output Heads ---
        # 1. Rotation Quaternion (4 val for w,x,y,z)
        self.rot_head = nn.Linear(256, 4)
        
        # 2. Residual Translation (3 val for x,y,z)
        self.trans_head = nn.Linear(256, 3)

    def forward(self, x):
        """
        x shape: (Batch, 3, Num_Points)
        """
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max Pooling (Symmetric Function) -> Ottiene feature globale
        x = torch.max(x, 2, keepdim=False)[0] # (Batch, 1024)
        
        # MLP
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        
        # Outputs
        # Normalized quaternion rotation
        pred_q = self.rot_head(x)
        pred_q = F.normalize(pred_q, p=2, dim=1)
        
        # Residual translation
        pred_t_res = self.trans_head(x)
        
        return pred_q, pred_t_res