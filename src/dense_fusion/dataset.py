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

class DenseFusionLineModDataset(Dataset):
    """
    LineMod Dataset adapted for DenseFusion (RGB + PointCloud).
    
    It converts 2D Depth maps into 3D Point Clouds via back-projection,
    loads the corresponding RGB crops, and generates pixel-point correspondence indices.
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
        num_points: int = 1024,
        object_ids: Optional[List[int]] = None,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        verbose: bool = True,
        resize_shape: tuple = (128, 128) # Nuova dimensione fissa per i crop RGB
    ):
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / 'data'
        self.split = split
        self.num_points = num_points
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.resize_shape = resize_shape 

        self.object_ids = object_ids if object_ids is not None else self.VALID_OBJECTS
        self.id_to_class = {obj_id: self.CLASS_NAMES[i] for i, obj_id in enumerate(self.VALID_OBJECTS)}
        
        # Standard LineMOD image dimensions
        self.input_standard_dimensions = (640, 480) 
        
        # Load cached configuration
        self.config = get_linemod_config(str(self.root_dir))
        
        # Build the dataset index
        self.samples = self._build_index()

        if verbose:
            print(f"✅ Loaded DenseFusionLineModDataset")
            print(f"   Split: {self.split} (Ratio: {self.train_ratio})")
            print(f"   Num Points: {self.num_points}")
            print(f"   Total samples: {len(self.samples)}")
            print(f"   RGB Resize Shape: {self.resize_shape}")

    def _build_index(self) -> List[Dict]:
        samples = []
        for obj_id in self.object_ids:
            try:
                gt_data = self.config.get_gt_data(obj_id)
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                continue

            all_img_ids = sorted([int(k) for k in gt_data.keys()])
            if not all_img_ids: continue

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
                annotations = gt_data.get(img_id_int) or gt_data.get(str(img_id_int)) or gt_data.get(f"{img_id_int:04d}")
                if not annotations: continue

                depth_path = obj_path / 'depth' / f"{img_id_int:04d}.png"
                if not depth_path.exists(): continue
                
                rgb_path = obj_path / 'rgb' / f"{img_id_int:04d}.png"
                if not rgb_path.exists(): continue

                for ann in annotations:
                    actula_obj_id = int(ann['obj_id'])
                    if actula_obj_id != obj_id: continue

                    rotation_matrix = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    translation_vector = np.array(ann['cam_t_m2c'])
                    quaternion_rotation = convert_rotation_to_quaternion(rotation_matrix)

                    x, y, w, h = map(int, ann['obj_bb'])
                    if w <= 0 or h <= 0: continue
                    
                    img_w, img_h = self.input_standard_dimensions
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img_w - x)
                    h = min(h, img_h - y)
                    if w <= 0 or h <= 0: continue

                    bbox = [x, y, w, h]
                    cam_info = info_data.get(img_id_int) or info_data.get(str(img_id_int)) or info_data.get(f"{img_id_int:04d}")
                    if cam_info is None: continue
                    cam_K = np.array(cam_info['cam_K']).reshape(3, 3)

                    sample = {
                        'object_id': actula_obj_id,
                        'class_idx': self.id_to_class[actula_obj_id],
                        'img_id': img_id_int,
                        'depth_path': depth_path,
                        'rgb_path': rgb_path,
                        'rotation': quaternion_rotation,
                        'translation': translation_vector / 1000.0,
                        'bbox': bbox,
                        'cam_K': cam_K
                    }
                    samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        depth_map = cv2.imread(str(sample['depth_path']), cv2.IMREAD_UNCHANGED)
        rgb_map = cv2.imread(str(sample['rgb_path']))
        
        if depth_map is None or rgb_map is None:
            return self.__getitem__((idx + 1) % len(self))
            
        rgb_map = cv2.cvtColor(rgb_map, cv2.COLOR_BGR2RGB)

        x, y, w, h = map(int, sample['bbox'])
        depth_crop = depth_map[y:y+h, x:x+w]
        rgb_crop = rgb_map[y:y+h, x:x+w]
        
        cam_K = sample['cam_K']
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        
        # Grid creation
        rows, cols = depth_crop.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        
        u_vals = c + x
        v_vals = r + y
        
        valid_mask = depth_crop > 0
        if not np.any(valid_mask):
             return self.__getitem__((idx + 1) % len(self))
        
        z_vals = depth_crop[valid_mask] / 1000.0
        u_vals = u_vals[valid_mask]
        v_vals = v_vals[valid_mask]
        
        # --- Handling Indices ---
        # Salviamo le coordinate U, V originali dei punti validi
        # prima di fare qualsiasi resize, così possiamo mappare
        # il punto 3D al pixel corretto nell'immagine resizata.
        
        # Coordinate nel crop originale (non globali)
        u_crop = c[valid_mask]
        v_crop = r[valid_mask]
        
        # Back projection
        x_vals = (u_vals - cx) * z_vals / fx
        y_vals = (v_vals - cy) * z_vals / fy
        points = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)
        
        # Sampling
        num_valid_points = points.shape[0]
        if num_valid_points >= self.num_points:
            choice_idx = np.random.choice(num_valid_points, self.num_points, replace=False)
        else:
            choice_idx = np.random.choice(num_valid_points, self.num_points, replace=True)
            
        points = points[choice_idx, :]
        
        # Prendiamo le coordinate pixel nel crop originale corrispondenti ai punti scelti
        u_chosen = u_crop[choice_idx] # X nel crop
        v_chosen = v_crop[choice_idx] # Y nel crop

        # --- RGB RESIZING (CRITICAL FIX) ---
        # 1. Resize the RGB crop to fixed dimensions
        orig_h, orig_w = rgb_crop.shape[:2]
        target_h, target_w = self.resize_shape
        rgb_resized = cv2.resize(rgb_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. Recalculate indices for the resized image
        # Scala le coordinate dei pixel selezionati
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        u_resized = (u_chosen * scale_x).astype(np.int32)
        v_resized = (v_chosen * scale_y).astype(np.int32)
        
        # Clip per sicurezza (evita indici fuori bordo per arrotondamenti)
        u_resized = np.clip(u_resized, 0, target_w - 1)
        v_resized = np.clip(v_resized, 0, target_h - 1)
        
        # Calcola gli indici flat per Gather nell'immagine resizata
        # Indice = y * width + x
        choose = v_resized * target_w + u_resized

        # Normalization
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        gt_translation = sample['translation']
        t_residual = gt_translation - centroid
        
        # Prepare Tensor
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0

        return {
            'points': torch.from_numpy(points_centered).T,
            'rgb': rgb_tensor,
            'choose': torch.from_numpy(choose).long(),
            'centroid': torch.from_numpy(centroid).float(),
            'rotation': torch.from_numpy(sample['rotation']).float(),
            't_residual': torch.from_numpy(t_residual).float(),
            'gt_translation': torch.from_numpy(gt_translation).float(),
            'object_id': sample['object_id'],
            'class_idx': self.id_to_class.get(sample['object_id'], 'unknown'),
            'img_id': sample['img_id'],
            'cam_K': torch.from_numpy(cam_K).float(),
            'img_path': str(sample['rgb_path'])
        }
    
    def get_image_path(self, idx: int):
        sample = self.samples[idx]
        return str(sample['rgb_path'])