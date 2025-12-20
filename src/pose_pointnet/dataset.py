import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional
from src.pose_rgb.pose_utils import convert_rotation_to_quaternion
from utils.linemod_config import get_linemod_config

class PointNetLineModDataset(Dataset):
    """
    LineMod Dataset for PointNet-based 6D Pose Estimation.
    
    It converts 2D Depth maps into 3D Point Clouds via back-projection
    and samples a fixed number of points (e.g., 1024).
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
        num_points: int = 1024,  # Fixed number of points required by PointNet
        object_ids: Optional[List[int]] = None,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ):
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / 'data'
        self.split = split
        self.num_points = num_points
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        self.object_ids = object_ids if object_ids is not None else self.VALID_OBJECTS
        self.id_to_class = {obj_id: self.CLASS_NAMES[i] for i, obj_id in enumerate(self.VALID_OBJECTS)}
        
        # Standard LineMOD image dimensions for boundary checks
        self.input_standard_dimensions = (640, 480) 
        
        # Load cached configuration
        self.config = get_linemod_config(str(self.root_dir))
        
        # Build the dataset index
        self.samples = self._build_index()

        print(f"✅ Loaded PointNetLineModDataset")
        print(f"   Split: {self.split} (Ratio: {self.train_ratio})")
        print(f"   Num Points: {self.num_points}")
        print(f"   Total samples: {len(self.samples)}")

    def _build_index(self) -> List[Dict]:
        """
        Builds the list of valid samples by scanning the dataset directory.
        It performs train/test splitting and filters out missing files.
        """
        samples = []

        for obj_id in self.object_ids:
            try:
                gt_data = self.config.get_gt_data(obj_id)
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                print(f"Warning: Data files not found for object {obj_id}")
                continue

            all_img_ids = sorted([int(k) for k in gt_data.keys()])
            if not all_img_ids:
                continue

            # Deterministic Shuffle
            rng = random.Random(self.random_seed)
            rng.shuffle(all_img_ids)

            # Train/Test Split
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
                # Robust key access (handle string vs int keys in YAML)
                annotations = gt_data.get(img_id_int) or gt_data.get(str(img_id_int)) or gt_data.get(f"{img_id_int:04d}")
                if not annotations:
                    continue

                # We only check for depth path as PointNet relies on geometry
                depth_path = obj_path / 'depth' / f"{img_id_int:04d}.png"
                if not depth_path.exists():
                    continue

                for ann in annotations:
                    actula_obj_id = int(ann['obj_id'])
                    # Ensure the annotation belongs to the current object
                    if actula_obj_id != obj_id:
                        continue

                    # Extract Pose
                    rotation_matrix = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    translation_vector = np.array(ann['cam_t_m2c'])
                    quaternion_rotation = convert_rotation_to_quaternion(rotation_matrix)

                    x, y, w, h = map(int, ann['obj_bb'])
                    
                    # Basic BBox Validation
                    if w <= 0 or h <= 0: continue
                    
                    # Clip bbox to image boundaries
                    img_w, img_h = self.input_standard_dimensions
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img_w - x)
                    h = min(h, img_h - y)
                    
                    if w <= 0 or h <= 0: continue

                    bbox = [x, y, w, h]

                    # Retrieve Camera Intrinsics
                    cam_info = info_data.get(img_id_int) or info_data.get(str(img_id_int)) or info_data.get(f"{img_id_int:04d}")
                    if cam_info is None: continue

                    cam_K = np.array(cam_info['cam_K']).reshape(3, 3)

                    sample = {
                        'object_id': actula_obj_id,
                        'class_idx': self.id_to_class[actula_obj_id],
                        'img_id': img_id_int,
                        'depth_path': depth_path,
                        'rotation': quaternion_rotation,
                        'translation': translation_vector / 1000.0, # Convert mm to METERS
                        'bbox': bbox,
                        'cam_K': cam_K
                    }
                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Loads the depth map, converts it to a 3D Point Cloud, 
        samples points, and computes the translation residual.
        """
        sample = self.samples[idx]
        
        # 1. Load Depth Map (usually 16-bit PNG in mm)
        depth_map = cv2.imread(str(sample['depth_path']), cv2.IMREAD_UNCHANGED)
        
        if depth_map is None:
            # Fallback: return the next item (rare case)
            return self.__getitem__((idx + 1) % len(self))

        # 2. Crop using Bounding Box
        x, y, w, h = map(int, sample['bbox'])
        depth_crop = depth_map[y:y+h, x:x+w]
        
        # 3. Back-projection (2D Pixels -> 3D Coordinates)
        cam_K = sample['cam_K']
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        
        # Create a grid of pixel coordinates (u, v) relative to the crop
        rows, cols = depth_crop.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Convert to global image coordinates
        u_vals = c + x
        v_vals = r + y
        
        # Create a mask for valid pixels (depth > 0)
        valid_mask = depth_crop > 0
        
        # Fallback if the crop contains no valid depth info
        if not np.any(valid_mask):
             return self.__getitem__((idx + 1) % len(self))
        
        # Extract valid values
        z_vals = depth_crop[valid_mask] / 1000.0 # Convert mm -> METERS
        u_vals = u_vals[valid_mask]
        v_vals = v_vals[valid_mask]
        
        # Apply Inverse Pinhole Camera model
        x_vals = (u_vals - cx) * z_vals / fx
        y_vals = (v_vals - cy) * z_vals / fy
        
        # Stack into a Point Cloud (N_valid, 3)
        points = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)
        
        # 4. Sampling (Ensure fixed number of points)
        num_valid_points = points.shape[0]
        
        if num_valid_points >= self.num_points:
            # If we have too many points, sample N without replacement
            choice_idx = np.random.choice(num_valid_points, self.num_points, replace=False)
        else:
            # If we have too few, sample N with replacement (padding)
            choice_idx = np.random.choice(num_valid_points, self.num_points, replace=True)
            
        points = points[choice_idx, :] # Shape: (num_points, 3)
        
        # 5. Normalization / Centering
        # PointNet learns the local shape. We subtract the centroid.
        # This makes the input invariant to the absolute position of the object.
        centroid = np.mean(points, axis=0) # (3,)
        points_centered = points - centroid
        
        # 6. Target Preparation
        # The network predicts:
        # A) Rotation (Quaternion)
        # B) Residual Translation = True Translation - Centroid
        gt_translation = sample['translation'] # (3,) in Meters
        t_residual = gt_translation - centroid
        
        # Return PyTorch Tensors
        return {
            # PointNet expects channels first: (3, N)
            'points': torch.from_numpy(points_centered).T, 
            'centroid': torch.from_numpy(centroid).float(), # Needed to reconstruct absolute translation
            'rotation': torch.from_numpy(sample['rotation']).float(),
            't_residual': torch.from_numpy(t_residual).float(),
            'gt_translation': torch.from_numpy(gt_translation).float(), # For debugging/eval
            'object_id': sample['object_id'],
            # Helper for visualization later
            'class_idx': self.id_to_class.get(sample['object_id'], 'unknown') ,
            'img_id': sample['img_id'],
            'cam_K': torch.from_numpy(cam_K).float()
        }
    
    def get_image_path(self, idx: int):
        """Get the image path for a given sample index."""
        sample = self.samples[idx]
        obj_folder = f"{sample['object_id']:02d}"
        img_path = self.data_dir / obj_folder / 'rgb' / f"{sample['img_id']:04d}.png"
        return str(img_path)