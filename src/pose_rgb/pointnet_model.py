import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetBackbone(nn.Module):
    """
    PointNet backbone per l'estrazione di feature globali da point cloud.
    
    Input: (B, N, 3) o (B, N, 6) dove N è il numero di punti
           - 3 canali: [x, y, z]
           - 6 canali: [x, y, z, r, g, b]
    
    Output: (B, 1024) feature globali
    """
    def __init__(self, input_channels=3, use_batch_norm=True):
        super(PointNetBackbone, self).__init__()
        
        # Shared MLPs (implementate come Conv1d per efficienza)
        # Ogni punto viene processato indipendentemente
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, C) point cloud dove C = 3 o 6
        Returns:
            global_feat: (B, 1024) feature globali
        """
        # PointNet lavora su (B, C, N) quindi traspongiamo
        x = x.transpose(2, 1)  # (B, N, C) -> (B, C, N)
        
        # Shared MLP layers
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        
        # Max pooling globale: (B, 1024, N) -> (B, 1024)
        # Questa è la chiave di PointNet: permutazione invariante
        global_feat = torch.max(x, 2)[0]
        
        return global_feat


class PointNetPose(nn.Module):
    """
    PointNet per 6D Pose Estimation.
    
    Predice rotation (quaternion) e translation GLOBALI da:
    - Point cloud LOCALE (coordinate relative al crop)
    - Bbox info (posizione percentuale del crop nell'immagine)
    """
    def __init__(self, input_channels=3, use_batch_norm=True):
        """
        Args:
            input_channels: 3 per [x,y,z] o 6 per [x,y,z,r,g,b]
            use_batch_norm: usare batch normalization
        """
        super(PointNetPose, self).__init__()
        
        # Backbone PointNet
        self.backbone = PointNetBackbone(input_channels, use_batch_norm)
        
        # Bbox info processor (4 valori: cx%, cy%, w%, h%)
        self.bbox_processor = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Feature fusion: 1024 (pointnet) + 128 (bbox) = 1152
        fusion_dim = 1024 + 128
        
        # Rotation Head (predice quaternion)
        self.rotation_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)  # Quaternion [w, x, y, z]
        )
        
        # Translation Head (predice [X, Y, Z] GLOBALE)
        self.translation_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # [tx, ty, tz] GLOBALE in metri
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Inizializza i pesi delle teste"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, point_cloud, bbox_info):
        """
        Forward pass.
        
        Args:
            point_cloud: (B, N, C) point cloud LOCALE
            bbox_info: (B, 4) bbox normalizzato [cx%, cy%, w%, h%]
        
        Returns:
            rotation: (B, 4) quaternion normalizzato GLOBALE
            translation: (B, 3) translation GLOBALE [x, y, z] in metri
        """
        # 1. Estrai feature globali dalla point cloud
        point_feat = self.backbone(point_cloud)
        
        # 2. Processa bbox info
        bbox_feat = self.bbox_processor(bbox_info)
        
        # 3. Fuse features
        fused_feat = torch.cat([point_feat, bbox_feat], dim=1)
        
        # 4. Predici rotation
        rotation = self.rotation_head(fused_feat)
        rotation = F.normalize(rotation, p=2, dim=1)  # Normalizza quaternion
        
        # 5. Predici translation GLOBALE
        translation = self.translation_head(fused_feat)
        
        return rotation, translation


class PointNetPoseWithRGBD(nn.Module):
    """
    Variante che combina PointNet con features RGB aggiuntive.
    Usa PointNet su point cloud [x,y,z,r,g,b] (6 canali).
    """
    def __init__(self):
        super(PointNetPoseWithRGBD, self).__init__()
        # PointNet con 6 canali input (xyz + rgb)
        self.pointnet = PointNetPose(input_channels=6, use_batch_norm=True)
    
    def forward(self, point_cloud_rgbd):
        """
        Args:
            point_cloud_rgbd: (B, N, 6) point cloud con RGB
                              [x, y, z, r, g, b]
        Returns:
            rotation: (B, 4) quaternion
            translation: (B, 3) translation in metri
        """
        return self.pointnet(point_cloud_rgbd)


# =============================================================================
# Utility Functions per creare Point Clouds dal dataset
# =============================================================================

def depth_to_point_cloud(depth_image, cam_K, rgb_image=None, subsample=1024):
    """
    Converte depth image + camera intrinsics in point cloud.
    
    Args:
        depth_image: (H, W) depth in mm
        cam_K: (3, 3) camera intrinsic matrix
        rgb_image: (H, W, 3) RGB image (opzionale)
        subsample: numero di punti da campionare (per efficienza)
    
    Returns:
        point_cloud: (subsample, 3) o (subsample, 6) se rgb_image è dato
    """
    import numpy as np
    
    H, W = depth_image.shape
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    
    # Crea mesh grid di coordinate pixel
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Maschera punti validi (depth > 0)
    valid_mask = depth_image > 0
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_image[valid_mask]
    
    # Back-projection: pixel -> 3D
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid
    
    # Point cloud XYZ
    points_xyz = np.stack([x, y, z], axis=1)  # (N, 3)
    
    # Aggiungi RGB se disponibile
    if rgb_image is not None:
        rgb_valid = rgb_image[valid_mask]  # (N, 3)
        points_xyzrgb = np.concatenate([points_xyz, rgb_valid / 255.0], axis=1)  # (N, 6)
        points = points_xyzrgb
    else:
        points = points_xyz
    
    # Subsample casuale per avere sempre lo stesso numero di punti
    if len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        points = points[indices]
    elif len(points) < subsample:
        # Pad con zeri se ci sono meno punti
        pad_size = subsample - len(points)
        points = np.vstack([points, np.zeros((pad_size, points.shape[1]))])
    
    return points.astype(np.float32)
