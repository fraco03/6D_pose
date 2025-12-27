import os
import cv2
import numpy as np
import yaml
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.pose_rgb.pose_utils import convert_rotation_to_quaternion
from utils.linemod_config import get_linemod_config

class LineModPoseDepthDataset(Dataset):
    """
    LineMod Dataset for 6D Pose Estimation with Depth Information.
    Supports optional loading of YOLO predicted bounding boxes.
    """

    VALID_OBJECTS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15] 

    CLASS_NAMES = [
        'ape', 'benchvise', 'camera', 'can', 'cat',
        'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
        'iron', 'lamp', 'phone'
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        object_ids: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        transform=None,
        normalize: bool = True,
        input_standard_dimensions: Tuple[int, int] = (640, 480),
        train_ratio: float = 0.8,
        random_seed: int = 42,
        verbose: bool = True,
        yolo_path: str = None  # New param for YOLO results
    ):
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / 'data'
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.yolo_path = Path(yolo_path) if yolo_path else None

        self.object_ids = object_ids if object_ids is not None else self.VALID_OBJECTS
        self.id_to_class = {obj_id: self.CLASS_NAMES[i] for i, obj_id in enumerate(self.VALID_OBJECTS)}
        self.input_standard_dimensions = input_standard_dimensions
        
        self.config = get_linemod_config(str(self.root_dir))
        
        self.samples = self._build_index()

        if verbose:
            split_print = self.train_ratio if self.split == 'train' else 1 - self.train_ratio
            print(f" Loaded LineModPoseDepthDataset")
            print(f"   Split: {self.split} (Ratio: {split_print:.2f})")
            print(f"   Objects: {self.object_ids} | YOLO Enabled: {self.yolo_path is not None}")
            print(f"   Total samples: {len(self.samples)}")

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def _build_index(self) -> List[Dict]:
        samples = []

        for obj_id in self.object_ids:
            # 1. Load GT and Camera Info
            try:
                gt_data = self.config.get_gt_data(obj_id)
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                print(f"Warning: Data files not found for object {obj_id}")
                continue

            # 2. Load YOLO Data (Optional)
            yolo_data = None
            if self.yolo_path:
                # Handle folders "1" vs "01" normalization
                yolo_folder = self.yolo_path / f"{obj_id:02d}"
                if not yolo_folder.exists():
                     yolo_folder = self.yolo_path / str(obj_id)
                
                yolo_file = yolo_folder / 'yolo.yml'
                if yolo_file.exists():
                    yolo_data = self._load_yaml(yolo_file)

            # Robust Key handling
            all_keys = list(gt_data.keys())
            all_img_ids = sorted([int(k) for k in all_keys])
            
            if not all_img_ids:
                continue

            rng = random.Random(self.random_seed)
            rng.shuffle(all_img_ids)

            split_idx = int(len(all_img_ids) * self.train_ratio)

            if self.split == 'train':
                selected_ids = all_img_ids[:split_idx]
            elif self.split == 'test' or self.split == 'val':
                selected_ids = all_img_ids[split_idx:]
            else:
                raise ValueError(f"Invalid split name: {self.split}")

            obj_folder = f"{obj_id:02d}"
            obj_path = self.data_dir / obj_folder

            for img_id_int in selected_ids:
                # Robust Keys
                key_str = str(img_id_int)
                key_pad = f"{img_id_int:04d}"
                
                # Get GT Annotations
                annotations = gt_data.get(img_id_int) or gt_data.get(key_str) or gt_data.get(key_pad)
                if not annotations: continue

                # Get YOLO Annotations (if available)
                y_anns = None
                if yolo_data:
                    y_anns = yolo_data.get(img_id_int) or yolo_data.get(key_str) or yolo_data.get(key_pad)

                img_path = obj_path / 'rgb' / f"{img_id_int:04d}.png"
                depth_path = obj_path / 'depth' / f"{img_id_int:04d}.png"

                # Check file existence
                if not img_path.exists() or not depth_path.exists():
                    continue

                for i, ann in enumerate(annotations):
                    actual_obj_id = int(ann['obj_id'])

                    # Only consider annotations for the current object
                    if actual_obj_id != obj_id: continue

                    # --- BOUNDING BOX SELECTION ---
                    bbox = ann['obj_bb'] # Default: GT
                    
                    # If YOLO is available and has an annotation for this index
                    if y_anns and i < len(y_anns):
                        # Use YOLO box (Fallback is handled in the .yml file generation phase)
                        bbox = y_anns[i]['obj_bb']

                    x, y, w, h = map(int, bbox)
                    
                    # BBox validity checks (YOLO predictions might be weird sometimes)
                    image_w, image_h = self.input_standard_dimensions
                    if w <= 0 or h <= 0: continue
                    
                    x0, y0 = x, y
                    x1, y1 = x + w, y + h
                    
                    # Strict boundary checks
                    if x1 <= x0 or y1 <= y0: continue
                    if not (0 <= x0 and 0 <= y0 and x1 <= image_w and y1 <= image_h):
                        continue

                    # Get camera info
                    cam_info = info_data.get(img_id_int) or info_data.get(key_str) or info_data.get(key_pad)
                    if cam_info is None: continue
                    
                    # Get raw pose
                    rotation_matrix = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    translation_vector = np.array(ann['cam_t_m2c'])
                    quaternion_rotation = convert_rotation_to_quaternion(rotation_matrix)
                    cam_K = np.array(cam_info['cam_K']).reshape(3, 3)

                    # Save metadata
                    sample = {
                        'object_id': actual_obj_id,
                        'class_idx': self.id_to_class[actual_obj_id],
                        'img_id': img_id_int,
                        'img_path': str(img_path),
                        'depth_path': str(depth_path),
                        'rotation': quaternion_rotation,
                        'translation': translation_vector / 1000.0, # m
                        'bbox': [x, y, w, h],  # This is the active BBox (GT or YOLO)
                        'cam_K': cam_K
                    }

                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 1. Load Image
        img = cv2.imread(sample['img_path'])
        if img is None:
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Load Depth Map
        depth_map = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED)
        
        # Depth error handling
        if depth_map is None:
            depth_map = np.zeros((self.input_standard_dimensions[1], self.input_standard_dimensions[0]), dtype=np.float32)
            Z_center = 0.0
        else:
            # 3. Load 3D Center from Depth Map
            # Uses the BBox selected in _build_index (GT or YOLO)
            x, y, w, h = map(int, sample['bbox'])
            img_h, img_w = depth_map.shape
            
            # Robust Center Calc (clip to image bounds)
            cx = min(max(x + w // 2, 0), img_w - 1)
            cy = min(max(y + h // 2, 0), img_h - 1)
            
            # Depth value at center (mm -> m)
            Z_center = depth_map[cy, cx] / 1000.0

        # Back-projection (Pinhole Model)
        cam_K = sample['cam_K']
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx_k, cy_k = cam_K[0, 2], cam_K[1, 2]

        # X = (u - cx) * Z / fx
        # Note: We use the geometric center of the bbox (cx, cy) to estimate translation
        X_center = (cx - cx_k) * Z_center / fx
        Y_center = (cy - cy_k) * Z_center / fy
        
        center_3d = np.array([X_center, Y_center, Z_center], dtype=np.float32)

        # 4. Crop & Resize Image
        x, y, w, h = map(int, sample['bbox'])
        
        # Safety clip
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.input_standard_dimensions[0], x + w)
        y1 = min(self.input_standard_dimensions[1], y + h)

        if x1 <= x0 or y1 <= y0:
            # Handle Empty Crop (rare but possible with bad YOLO pred)
            cropped_img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.float32)
            cropped_depth = np.zeros(self.image_size, dtype=np.float32)
        else:
            cropped_img = img[y0:y1, x0:x1]
            cropped_img = cv2.resize(cropped_img, self.image_size)

            # 5. Crop & Resize Depth (Channel 4)
            cropped_depth = depth_map[y0:y1, x0:x1]
            cropped_depth = np.clip(cropped_depth, 0.1, 5000.0) # mm
            cropped_depth = cv2.resize(cropped_depth, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Normalization steps...
        if self.normalize:
            cropped_img = cropped_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            cropped_img = (cropped_img - mean) / std

        if self.transform:
            cropped_img = self.transform(cropped_img)

        # Tensor conversion
        img_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1).float()
        
        # Depth tensor prep (Normalized to 0-1 range approx for CNN stability)
        depth_tensor = torch.from_numpy(cropped_depth).float()
        depth_max = 2000.0 # 2m max depth assumption
        depth_tensor = torch.clamp(depth_tensor / depth_max, 0.0, 1.0).unsqueeze(0)

        return {
            'image': img_tensor,
            'depth': depth_tensor,
            'img_id': sample['img_id'],
            'img_path': sample['img_path'],
            'rotation': torch.from_numpy(sample['rotation']).float(),
            'translation': torch.from_numpy(sample['translation']).float(),
            '3D_center': torch.from_numpy(center_3d).float(), 
            'object_id': sample['object_id'],
            'class_idx': sample['class_idx'],
            'cam_K': torch.from_numpy(sample['cam_K']).float(),
            'bbox': torch.tensor(sample['bbox'], dtype=torch.float32),
            'original_width': self.input_standard_dimensions[0],
            'original_height': self.input_standard_dimensions[1]
        }