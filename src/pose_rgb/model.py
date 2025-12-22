import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ResNetRotation(nn.Module):
    """
    ResNet-based network dedicated to 6D Rotation Estimation.
    
    Architecture based on the project requirements:
    - Backbone: ResNet-50 (pretrained on ImageNet)
    - Head: Regression layers predicting a unit quaternion.
    """

    def __init__(self, freeze_backbone: bool = True):
        """
        Args:
            freeze_backbone (bool): If True, prevents the ResNet weights from being updated 
                                    during the first phase of training. Recommended to avoid 
                                    destroying pretrained features.
        """
        super(ResNetRotation, self).__init__()

        # 1. Load Pretrained ResNet-50
        # We use standard ImageNet weights as the initialization
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)

        # 2. Feature Extractor
        # We remove the final classification layer (fc) but keep the Global Average Pooling.
        # This takes the input image and outputs a vector of size 2048.
        # list(resnet.children())[:-1] returns all layers except the last one.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet-50 feature dimension is always 2048 before the final layer
        feature_dim = resnet.fc.in_features 

        # 3. Freeze Backbone (Optional but Recommended)
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            print("🔒 ResNet backbone frozen.")

        # 4. Regression Head
        # Maps the 2048 features to a 4D vector (Quaternion).
        # We add a hidden layer (512 or 1024) to learn non-linear relationships.
        self.rot_head = nn.Sequential(
            nn.Flatten(),                # Flatten (B, 2048, 1, 1) -> (B, 2048)
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),                   # Non-linearity
            nn.Dropout(p=0.2),           # Dropout to prevent overfitting
            nn.Linear(1024, 4)           # Output: 4 values (w, x, y, z)
        )

        # Initialize the head weights (He initialization is good for ReLU)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new layers with proper scaling."""
        for m in self.rot_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
            """
            Forward pass.

            Args:
                x (torch.Tensor): Input images batch. Shape (Batch_Size, 3, 224, 224).
            
            Returns:
                torch.Tensor: Normalized quaternions. Shape (Batch_Size, 4).
            """
            # 1. Extract visual features
            x = self.features(x)  # Shape: (B, 2048, 1, 1)
            
            # 2. Regress quaternion
            q = self.rot_head(x)  # Shape: (B, 4)

            # 3. Normalize Quaternion [Important!]
            # A rotation quaternion must have unit length (magnitude = 1).
            # We enforce this constraint directly in the network output.
            q = F.normalize(q, p=2, dim=1)

            return q
        

class TranslationNet(nn.Module):
    """
    Custom Lightweight Network for 3D Translation Estimation.
    
    It predicts the 3D position by combining:
    1. Visual features from the RGB crop (to understand local structure/offset).
    2. Geometric features from the Bounding Box (to understand scale/depth).
    
    Output:
    - [delta_x, delta_y, z]
      * delta_x, delta_y: Pixel offsets from the crop center to the object centroid.
      * z: Absolute depth in meters.
    """

    def __init__(self):
        super(TranslationNet, self).__init__()

        # --- 1. VISUAL BRANCH (Custom CNN) ---
        # Extracts features from the resized RGB crop (224x224).
        # Since we train from scratch, we use a deeper architecture (4 blocks)
        # to learn hierarchical features (edges -> shapes -> object parts).
        self.conv_layers = nn.Sequential(
            # Block 1: Input (B, 3, 224, 224) -> Output (B, 32, 56, 56)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            # Block 2: -> (B, 64, 28, 28)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 3: -> (B, 128, 14, 14)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 4: -> (B, 256, 7, 7)
            # Added depth to capture more complex visual patterns
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Final pooling to force a fixed size (4x4 spatial dims)
            # We use (4, 4) instead of (1, 1) to preserve spatial information
            # Output: (B, 256, 4, 4)
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        
        # Flattened visual vector size: 256 channels * 4 * 4 = 4096
        self.visual_flat_dim = 256 * 4 * 4

        # --- 2. GEOMETRIC BRANCH (MLP) ---
        # Processes the normalized BBox coordinates: [center_x%, center_y%, w%, h%]
        # This branch is crucial for estimating 'z' (depth) based on scale.
        self.geo_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        geo_dim = 128

        # --- 3. FUSION & REGRESSION HEAD ---
        # Concatenates Visual (4096) + Geometric (128) features.
        self.regressor = nn.Sequential(
            nn.Linear(self.visual_flat_dim + geo_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4), # Increased Dropout to 0.4 for training from scratch
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # Output: [delta_x, delta_y, z]
        )

        # Initialize weights for better training stability
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using Kaiming Normal (He init) for Conv/Linear layers 
        followed by ReLU.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, bbox_norm):
        """
        Forward pass.
        
        Args:
            img (torch.Tensor): RGB image crops. Shape (Batch, 3, 224, 224).
            bbox_norm (torch.Tensor): Normalized bbox info. Shape (Batch, 4).
                                      Format: [cx_perc, cy_perc, w_perc, h_perc]
        
        Returns:
            torch.Tensor: Prediction vector (Batch, 3) containing [dx, dy, z].
        """
        # 1. Process Visual Branch
        v = self.conv_layers(img)
        v = v.view(v.size(0), -1) # Flatten (B, 4096)
        
        # 2. Process Geometric Branch
        g = self.geo_fc(bbox_norm) # (B, 128)
        
        # 3. Feature Fusion
        combined = torch.cat((v, g), dim=1) # Shape (Batch, 4224)
        
        # 4. Prediction
        pred = self.regressor(combined)
        
        return pred
    

class UnifiedPoseNet(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(UnifiedPoseNet, self).__init__()
        
        # 1. Backbone: ResNet50
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        
        # Rimuoviamo il FC layer finale e l'avgpool, teniamo tutto fino all'ultimo blocco conv
        # Output shape: (Batch, 2048, 7, 7) per immagini 224x224
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dimensione features ResNet50
        self.feature_dim = 2048 
        
        # 2. Fusion Layer
        # Concateniamo le features visive (2048) con le info geometriche della BBox (4)
        # BBox info: [cx_perc, cy_perc, w_perc, h_perc]
        self.fusion_dim = self.feature_dim + 4
        
        # 3. Rotation Head (Quaternion)
        self.rot_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 4)  # Quaternion (w, x, y, z)
        )
        
        # 4. Translation Head (x, y, z)
        # Qui la BBox info è CRUCIALE per stimare la Z (profondità)
        self.trans_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 3)  # Translation (tx, ty, tz) in metri
        )
        
        self._init_weights()

    def _init_weights(self):
        # Inizializzazione per i layer lineari aggiunti
        for m in self.rot_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        for m in self.trans_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, img, bbox_info):
        """
        img: (B, 3, 224, 224)
        bbox_info: (B, 4) -> [cx, cy, w, h] normalizzati
        """
        # Feature Extraction
        x = self.backbone(img)      # (B, 2048, 7, 7)
        x = self.avgpool(x)         # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)     # (B, 2048)
        
        # Feature Fusion (Visual + Geometric)
        x = torch.cat((x, bbox_info), dim=1) # (B, 2052)
        
        # Heads
        rot = self.rot_head(x)
        trans = self.trans_head(x)
        
        # Normalizzazione Quaternione
        rot = F.normalize(rot, p=2, dim=1)
        
        return rot, trans