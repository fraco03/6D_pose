import os
import random
import cv2
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .pose_utils import convert_rotation_to_quaternion
from utils.linemod_config import get_linemod_config

class LineModPoseDataset(Dataset):
    """
    LineMod Dataset for 6D Pose Estimation.
    Supports loading YOLO predicted bounding boxes via 'yolo_path'.
    """

    VALID_OBJECTS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    CLASS_NAMES = [
        'ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 
        'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone'
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
            print(f" Loaded LineModPoseDataset | Split: {self.split}")
            print(f" Objects: {self.object_ids} | YOLO Enabled: {self.yolo_path is not None}")
            print(f" Samples: {len(self.samples)}")

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def _build_index(self) -> List[Dict]:
        samples = []

        for obj_id in self.object_ids:
            # 1. Load GT Data
            try:
                gt_data = self.config.get_gt_data(obj_id)
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                print(f"Warning: GT files not found for object {obj_id}")
                continue

            # 2. Load YOLO Data (Optional)
            yolo_data = None
            if self.yolo_path:
                # Handle potential folder naming mismatch ("1" vs "01")
                yolo_folder = self.yolo_path / f"{obj_id:02d}"
                if not yolo_folder.exists():
                    yolo_folder = self.yolo_path / str(obj_id)
                
                yolo_file = yolo_folder / 'yolo.yml'
                if yolo_file.exists():
                    yolo_data = self._load_yaml(yolo_file)

            # 3. Process Keys and Split
            all_keys = sorted([int(k) for k in gt_data.keys()])
            if not all_keys: continue

            rng = random.Random(self.random_seed)
            rng.shuffle(all_keys)

            split_idx = int(len(all_keys) * self.train_ratio)
            selected_ids = all_keys[:split_idx] if self.split == 'train' else all_keys[split_idx:]

            obj_path = self.data_dir / f"{obj_id:02d}"

            # 4. Create Samples
            for img_id_int in selected_ids:
                # Robust key lookup
                key_str = str(img_id_int)
                key_pad = f"{img_id_int:04d}"
                
                anns = gt_data.get(img_id_int) or gt_data.get(key_str) or gt_data.get(key_pad)
                if not anns: continue

                # Retrieve YOLO anns for this frame
                y_anns = None
                if yolo_data:
                    y_anns = yolo_data.get(img_id_int) or yolo_data.get(key_str) or yolo_data.get(key_pad)

                img_path = obj_path / 'rgb' / f"{img_id_int:04d}.png"
                if not img_path.exists():
                    img_path = obj_path / f"{img_id_int:04d}.png" # Fallback
                    if not img_path.exists(): continue

                for i, ann in enumerate(anns):
                    if int(ann['obj_id']) != obj_id: continue

                    # --- Select Bounding Box ---
                    bbox = ann['obj_bb'] # Default GT
                    
                    # If YOLO data exists for this frame/object, use it
                    if y_anns and i < len(y_anns):
                        # The yolo.yml generator already handled the fallback logic (is_yolo=False if detection failed)
                        # so we just take what's in the file.
                        bbox = y_anns[i]['obj_bb']

                    # --- Validation ---
                    x, y, w, h = map(int, bbox)
                    if w <= 0 or h <= 0: continue
                    
                    # Boundary checks
                    std_w, std_h = self.input_standard_dimensions
                    if not (0 <= x and 0 <= y and (x + w) <= std_w and (y + h) <= std_h):
                        continue

                    # Camera info
                    cam_info = info_data.get(img_id_int) or info_data.get(key_str) or info_data.get(key_pad)
                    if not cam_info: continue

                    # Prepare Sample
                    rot_mx = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    
                    sample = {
                        'object_id': obj_id,
                        'class_idx': self.id_to_class[obj_id],
                        'img_id': img_id_int,
                        'img_path': img_path,
                        'rotation': convert_rotation_to_quaternion(rot_mx),
                        'translation': np.array(ann['cam_t_m2c']) / 1000.0,
                        'bbox': bbox,  # Can be GT or YOLO
                        'cam_K': np.array(cam_info['cam_K']).reshape(3, 3)
                    }
                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load Image
        img = cv2.imread(str(sample['img_path']))
        if img is None:
            # Return empty black image on failure
            return self._get_empty_sample(sample)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = map(int, sample['bbox'])
        
        # Robust Cropping
        H, W = img.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)

        if x1 <= x0 or y1 <= y0:
            return self._get_empty_sample(sample)

        cropped_img = img[y0:y1, x0:x1]
        if cropped_img.size == 0:
             return self._get_empty_sample(sample)
             
        cropped_img = cv2.resize(cropped_img, self.image_size)

        # Normalization
        if self.normalize:
            cropped_img = cropped_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            cropped_img = (cropped_img - mean) / std

        if self.transform:
            cropped_img = self.transform(cropped_img)

        # Prepare Tensors
        crop_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1).float()
        
        # BBox Info (Normalized relative to standard input dims)
        std_w, std_h = self.input_standard_dimensions
        cx, cy = x + w / 2.0, y + h / 2.0
        
        bbox_info = torch.tensor([
            cx / float(std_w), cy / float(std_h),
            w / float(std_w), h / float(std_h)
        ], dtype=torch.float32)
        
        return {
            'image': crop_tensor,
            'rotation': torch.from_numpy(sample['rotation']).float(),
            'translation': torch.from_numpy(sample['translation']).float(),
            'object_id': sample['object_id'],
            'class_idx': sample['class_idx'],
            'cam_K': torch.from_numpy(sample['cam_K']).float(),
            'img_id': sample['img_id'],
            'bbox_info': bbox_info,
            'bbox_center': torch.tensor([cx, cy], dtype=torch.float32),
            'original_width': std_w,
            'original_height': std_h,
            'img_path': str(sample['img_path'])
        }

    def _get_empty_sample(self, sample):
        # Fallback for corrupted images/crops
        empty_img = torch.zeros((3, self.image_size[1], self.image_size[0]), dtype=torch.float32)
        std_w, std_h = self.input_standard_dimensions
        return {
            'image': empty_img,
            'rotation': torch.from_numpy(sample['rotation']).float(),
            'translation': torch.from_numpy(sample['translation']).float(),
            'object_id': sample['object_id'],
            'class_idx': sample['class_idx'],
            'cam_K': torch.from_numpy(sample['cam_K']).float(),
            'img_id': sample['img_id'],
            'bbox_info': torch.zeros(4, dtype=torch.float32),
            'bbox_center': torch.zeros(2, dtype=torch.float32),
            'original_width': std_w,
            'original_height': std_h,
            'img_path': str(sample['img_path'])
        }

    def get_class_name(self, class_idx: int) -> str:
        return self.id_to_class.get(class_idx, "Unknown")