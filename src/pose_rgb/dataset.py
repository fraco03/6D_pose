import os
import random
import cv2
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .pose_utils import convert_rotation_to_quaternion
from utils.linemod_config import get_linemod_config

class LineModPoseDataset(Dataset):
    """
    LineMod Dataset for 6D Pose Estimation

    Returns cropped object images and corresponding poses.
    """

    VALID_OBJECTS = [
        1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15
    ]  # object id 3, 7 not available

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
        transform = None,
        normalize: bool = True,
        input_standard_dimensions: Tuple[int, int] = (640, 480),
        train_ratio: float = 0.8,  # Percentage of data for training
        random_seed: int = 42,      # Fixed seed for reproducibility,
        verbose: bool = True
    ):
        """
        Args:
            root_dir (str): Path to the LineMod preprocessed dataset.
            split (str): 'train' or 'test' split.
            object_ids (List[int], optional): List of object IDs to include. If None, include all.
            image_size (Tuple[int, int]): Size to which images are resized.
            transform: Optional transform to be applied on a sample.
            normalize (bool): Whether to imagenet normalization.
        """

        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / 'data'
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        self.object_ids = object_ids if object_ids is not None else self.VALID_OBJECTS
        self.id_to_class = {obj_id: self.CLASS_NAMES[i] for i, obj_id in enumerate(self.VALID_OBJECTS)}
        self.input_standard_dimensions = input_standard_dimensions
        
        self.config = get_linemod_config(str(self.root_dir))
        
        self.samples = self._build_index()
        if verbose:
            print_ratio = self.train_ratio if self.split == 'train' else 1 - self.train_ratio
            print(f" Loaded LineModPoseDataset")
            print(f"   Split: {self.split} (Ratio: {print_ratio:.2f})")
            print(f"   Objects: {self.object_ids}")
            print(f"   Total samples: {len(self.samples)}")

    def _build_index(self) -> List[Dict]:
        samples = []

        for obj_id in self.object_ids:
            # 1. Load GT data
            try:
                gt_data = self.config.get_gt_data(obj_id)
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                print(f"Warning: Data files not found for object {obj_id}")
                continue

            # 2. Retrieve ALL valid image IDs
            # Robust key handling: keys might be strings or ints in the loaded dict
            all_keys = list(gt_data.keys())
            all_img_ids = sorted([int(k) for k in all_keys])
            
            if not all_img_ids:
                continue

            # 3. Deterministic Shuffle
            rng = random.Random(self.random_seed)
            rng.shuffle(all_img_ids)

            # 4. Split logic
            split_idx = int(len(all_img_ids) * self.train_ratio)

            if self.split == 'train':
                selected_ids = all_img_ids[:split_idx]
            elif self.split == 'test' or self.split == 'val':
                selected_ids = all_img_ids[split_idx:]
            else:
                raise ValueError(f"Invalid split name: {self.split}")

            obj_folder = f"{obj_id:02d}"
            obj_path = self.data_dir / obj_folder

            # 5. Load samples
            for img_id_int in selected_ids:
                
                # --- FIX 1: Accesso robusto alle chiavi (come nel Depth Dataset) ---
                annotations = gt_data.get(img_id_int) or gt_data.get(str(img_id_int)) or gt_data.get(f"{img_id_int:04d}")
                if not annotations:
                    continue

                # Paths
                img_path = obj_path / 'rgb' / f"{img_id_int:04d}.png"
                
                # Se usi il depth dataset accoppiato, dovresti controllare anche l'esistenza del depth qui
                # depth_path = obj_path / 'depth' / f"{img_id_int:04d}.png"
                
                if not img_path.exists(): 
                    continue

                for ann in annotations:
                    actual_obj_id = int(ann['obj_id'])

                    # --- FIX 2: Filtro fondamentale dell'ID ---
                    # Ignoriamo le annotazioni di altri oggetti presenti nella stessa foto
                    if actual_obj_id != obj_id:
                        continue

                    rotation_matrix = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    translation_vector = np.array(ann['cam_t_m2c'])
                    quaternion_rotation = convert_rotation_to_quaternion(rotation_matrix)

                    x, y, w, h = map(int, ann['obj_bb'])
                    
                    # --- FIX 3: Validazione BBox Identica ---
                    image_w, image_h = self.input_standard_dimensions

                    # Skip invalid size
                    if w <= 0 or h <= 0: continue
                    
                    x0, y0 = x, y
                    x1, y1 = x + w, y + h
                        
                    # invalid bb coordinates
                    if x1 <= x0 or y1 <= y0: continue
                    
                    # Strict boundary check (deve essere identico nell'altro dataset)
                    if not (0 <= x0 and 0 <= y0 and x1 <= image_w and y1 <= image_h):
                        continue
                    
                    # Recuperiamo le info camera (Gestione robusta chiavi)
                    cam_info = info_data.get(img_id_int) or info_data.get(str(img_id_int)) or info_data.get(f"{img_id_int:04d}")
                    if cam_info is None:
                        continue

                    sample = {
                        'object_id': actual_obj_id,
                        'class_idx': self.id_to_class[actual_obj_id],
                        'img_id': img_id_int,
                        'img_path': img_path,
                        'rotation': quaternion_rotation,
                        'translation': translation_vector/1000.0,
                        'bbox': ann['obj_bb'],
                        'cam_K': np.array(cam_info['cam_K']).reshape(3, 3)
                    }

                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image
        img = cv2.imread(str(sample['img_path']))
        if img is None:  # Handle case where image file might be corrupted or missing
            print(f"Warning: Could not load image from {sample['img_path']}. Returning black image.")
            cropped_img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x, y, w, h = tuple(map(int, sample['bbox']))
            # Crop using validated bounding box (already checked in _build_index)
            H, W = img.shape[:2]
            # clip bbox to image bounds
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x + w)
            y1 = min(H, y + h)

            if x1 <= x0 or y1 <= y0:
                print(f"Warning: Invalid/empty crop for {sample['img_path']} bbox {sample['bbox']}. Returning black image.")
                cropped_img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            else:
                cropped_img = img[y0:y1, x0:x1]
                cropped_img = cv2.resize(cropped_img, self.image_size)

                # (optional) if you want cx/cy to match the clipped box:
                x, y, w, h = x0, y0, (x1 - x0), (y1 - y0)

            # In rare cases if the crop is empty due to unexpected data, fallback to black
            if cropped_img.size == 0:
                print(f"Warning: Empty crop for image {sample['img_path']} with bbox {sample['bbox']}. Returning black image.")
                cropped_img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

                
            # Resize to desired size
            cropped_img = cv2.resize(cropped_img, self.image_size)

        if self.normalize:
            # Standard ImageNet normalization
            cropped_img = cropped_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            cropped_img = (cropped_img - mean) / std

        if self.transform:
            cropped_img = self.transform(cropped_img)

        # From (H, W, 3) to (3, H, W)
        crop_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1).float()
        
        
        # Calculate Center (in pixels)
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        # Create Normalized Tensor [cx%, cy%, w%, h%] for TranslationNet
        # We use the standard dimensions stored in self.input_standard_dimensions (640, 480)
        std_w, std_h = self.input_standard_dimensions
        
        bbox_info = torch.tensor([
            cx / float(std_w),
            cy / float(std_h),
            w  / float(std_w),
            h  / float(std_h)
        ], dtype=torch.float32)
        
        # Create Center Tensor [cx, cy] for Loss Calculation
        bbox_center = torch.tensor([cx, cy], dtype=torch.float32)
        return {
            'image': crop_tensor,                                               # (3, 224, 224)
            'rotation': torch.from_numpy(sample['rotation']).float(),           # (4,) [w,x,y,z]
            'translation': torch.from_numpy(sample['translation']).float(),     # (3,)
            'object_id': sample['object_id'],                                   # int
            'class_idx': sample['class_idx'],                                   # int
            'cam_K': torch.from_numpy(sample['cam_K']).float(),                 # (3, 3)
            'img_id': sample['img_id'],                                          # int
            'bbox_info': bbox_info,       # Input for the Network
            'bbox_center': bbox_center,    # Helper for the Loss function
            'original_width': self.input_standard_dimensions[0],
            'original_height': self.input_standard_dimensions[1],
            'img_path': str(sample['img_path'])
        }

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from class index"""
        return self.id_to_class.get(class_idx, "Unknown")



            


        