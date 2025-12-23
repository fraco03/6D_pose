import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
    

class FusionPoseModel(nn.Module):
    def __init__(self, num_points=1024):
        super(FusionPoseModel, self).__init__()
        
        # --- RAMO 1: GEOMETRIA (PointNet Encoder) ---
        # Input: (Batch, 3, Num_Points)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # --- RAMO 2: COLORE (CNN Encoder - ResNet18) ---
        # Input: (Batch, 3, 224, 224) -> Feature Vector
        # Usiamo ResNet18 pre-addestrata (leggera e veloce)
        resnet = models.resnet18(pretrained=True)
        # Rimuoviamo l'ultimo livello Fully Connected (fc) per avere le feature pure
        # L'output prima della fc è (Batch, 512, 1, 1) dopo l'AvgPool
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Nota: ResNet18 outputta 512 canali
        
        # --- FUSIONE ---
        # Feature PointNet (1024) + Feature RGB (512) = 1536
        self.fc1 = nn.Linear(1024 + 512, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        
        # --- OUTPUT HEADS ---
        # Rotazione (Quaternione)
        self.rot_head = nn.Linear(256, 4)
        # Traslazione (Residuo)
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
        
        # 3. Fusion (Concatenazione)
        # (Batch, 1024+512) -> (Batch, 1536)
        global_feat = torch.cat([feat_geo, feat_rgb], dim=1)
        
        # 4. Regressor MLP
        x = F.relu(self.bn_fc1(self.fc1(global_feat)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        
        # 5. Predictions
        pred_q = self.rot_head(x)
        pred_q = F.normalize(pred_q, p=2, dim=1) # Quaternione unitario
        
        pred_t_res = self.trans_head(x)
        
        return pred_q, pred_t_res