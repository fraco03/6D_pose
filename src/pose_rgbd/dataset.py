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
    
    Modified to support automatic random train/test splitting instead of reading 
    pre-defined text files.
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
        transform=None,
        normalize: bool = True,
        input_standard_dimensions: Tuple[int, int] = (640, 480),
        train_ratio: float = 0.8,  # Percentage of data for training
        random_seed: int = 42      # Fixed seed for reproducibility
    ):
        """
        Args:
            root_dir (str): Path to the LineMod preprocessed dataset.
            split (str): 'train' or 'test' split.
            object_ids (List[int], optional): List of object IDs to include.
            image_size (Tuple[int, int]): Size to which images are resized.
            transform: Optional transform to be applied on a sample.
            normalize (bool): Whether to apply ImageNet normalization.
            train_ratio (float): Ratio for splitting data (default 0.8).
            random_seed (int): Seed to ensure the split remains constant.
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
        
        # Get cached config instance
        self.config = get_linemod_config(str(self.root_dir))
        
        self.samples = self._build_index()

        print(f" Loaded LineModPoseDepthDataset")
        print(f"   Split: {self.split} (Ratio: {self.train_ratio})")
        print(f"   Objects: {self.object_ids}")
        print(f"   Total samples: {len(self.samples)}")

    def _build_index(self) -> List[Dict]:
        samples = []

        for obj_id in self.object_ids:
            # 1. Load GT data and Camera info using cached config
            try:
                gt_data = self.config.get_gt_data(obj_id)
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                print(f"Warning: Data files not found for object {obj_id}")
                continue

            # 2. Retrieve ALL valid image IDs available for this object
            # We use the keys from gt_data as the master list.
            all_img_ids = sorted([int(k) for k in gt_data.keys()])
            
            if not all_img_ids:
                continue

            # 3. Apply Deterministic Shuffle
            # Using a local Random instance prevents affecting the global random state
            rng = random.Random(self.random_seed)
            rng.shuffle(all_img_ids)

            # 4. Calculate the split index
            split_idx = int(len(all_img_ids) * self.train_ratio)

            # 5. Select IDs based on the requested split
            if self.split == 'train':
                selected_ids = all_img_ids[:split_idx]
            elif self.split == 'test' or self.split == 'val':
                selected_ids = all_img_ids[split_idx:]
            else:
                raise ValueError(f"Invalid split name: {self.split}. Use 'train' or 'test'.")

            obj_folder = f"{obj_id:02d}"
            obj_path = self.data_dir / obj_folder

            # 6. Load samples for the selected IDs
            for img_id_int in selected_ids:
                
                # Handle potential key format differences (int vs string)
                annotations = gt_data.get(img_id_int) or gt_data.get(str(img_id_int)) or gt_data.get(f"{img_id_int:04d}")
                
                if not annotations:
                    continue

                # Construct file paths
                img_path = obj_path / 'rgb' / f"{img_id_int:04d}.png"
                depth_path = obj_path / 'depth' / f"{img_id_int:04d}.png"

                if not img_path.exists() or not depth_path.exists():
                    continue # Skip if image or depth does not exist

                for ann in annotations:
                    actula_obj_id = int(ann['obj_id'])

                    # Safety check: ensure the annotation belongs to the current object
                    if actula_obj_id != obj_id:
                        continue

                    rotation_matrix = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    translation_vector = np.array(ann['cam_t_m2c'])
                    quaternion_rotation = convert_rotation_to_quaternion(rotation_matrix)

                    x, y, w, h = map(int, ann['obj_bb'])
                    bbox = [x, y, w, h]

                    # Validate bbox
                    image_w, image_h = self.input_standard_dimensions

                    if w <= 0 or h <= 0:
                        continue

                    x0, y0 = x, y
                    x1, y1 = x + w, y + h

                    if x1 <= x0 or y1 <= y0:
                        continue

                    if not (0 <= x0 and 0 <= y0 and x1 <= image_w and y1 <= image_h):
                        continue

                    # Calculate depth at the center of the bounding box
                    # We load the depth map here to compute the 3D center
                    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if depth_map is None:
                        print(f"Warning: Could not load depth map from {depth_path}. Skipping sample.")
                        continue

                    # Clamp coordinates to image boundaries
                    cx, cy = x + w // 2, y + h // 2
                    cx = min(max(cx, 0), image_w - 1)
                    cy = min(max(cy, 0), image_h - 1)

                    depth_center = depth_map[cy, cx] / 1000.0  # Convert mm to meters

                    # Retrieve camera info for this image
                    cam_info = info_data.get(img_id_int) or info_data.get(str(img_id_int)) or info_data.get(f"{img_id_int:04d}")
                    
                    if cam_info is None:
                        continue

                    # Calculate 3D coordinates using inverse pinhole model
                    cam_K = np.array(cam_info['cam_K']).reshape(3, 3)
                    fx, fy = cam_K[0, 0], cam_K[1, 1]
                    cx_k, cy_k = cam_K[0, 2], cam_K[1, 2]

                    X = (cx - cx_k) * depth_center / fx
                    Y = (cy - cy_k) * depth_center / fy
                    Z = depth_center

                    sample = {
                        'object_id': actula_obj_id,
                        'class_idx': self.id_to_class[actula_obj_id],
                        'img_id': img_id_int,
                        'img_path': img_path,
                        'depth_path': depth_path,
                        'rotation': quaternion_rotation,
                        'translation': translation_vector / 1000.0,  # Convert mm to meters
                        'bbox': ann['obj_bb'],  # [x, y, w, h]
                        'cam_K': cam_K,
                        '3D_center': np.array([X, Y, Z], dtype=np.float32)
                    }

                    samples.append(sample)

        return samples
    

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image
        img = cv2.imread(str(sample['img_path']))
        if img is None:
            print(f"Warning: Could not load image from {sample['img_path']}. Returning black image.")
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load depth map
        depth_map = cv2.imread(str(sample['depth_path']), cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            print(f"Warning: Could not load depth map from {sample['depth_path']}. Returning zero depth.")
            depth_map = np.zeros((self.input_standard_dimensions[1], self.input_standard_dimensions[0]), dtype=np.float32)

        # Crop using bounding box
        x, y, w, h = map(int, sample['bbox'])
        
        # Ensure crop coordinates are within bounds
        img_h, img_w = depth_map.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        cropped_img = img[y:y + h, x:x + w]
        cropped_depth = depth_map[y:y + h, x:x + w]
        
        # Clip depth values to a reasonable range
        cropped_depth = np.clip(cropped_depth, 0.1, 5000.0) # Clip assuming mm input first

        # Resize to desired size
        cropped_img = cv2.resize(cropped_img, self.image_size)
        cropped_depth = cv2.resize(cropped_depth, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.normalize:
            cropped_img = cropped_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            cropped_img = (cropped_img - mean) / std

        if self.transform:
            cropped_img = self.transform(cropped_img)

        img_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1).float()
        depth_tensor = torch.from_numpy(cropped_depth).float()

        # Normalize depth map: convert mm to meters and scale
        depth_max = 2000.0 # Define a max depth (e.g., 2 meters = 2000mm)
        depth_tensor = torch.clamp(depth_tensor / depth_max, 0.0, 1.0).unsqueeze(0)  # Normalize and add channel dim
        
        return {
            'image': img_tensor,
            'depth': depth_tensor,
            'img_id': sample['img_id'],
            'img_path': str(sample['img_path']),
            'rotation': torch.from_numpy(sample['rotation']).float(),
            'translation': torch.from_numpy(sample['translation']).float(),
            '3D_center': torch.from_numpy(sample['3D_center']).float(),
            'object_id': sample['object_id'],
            'class_idx': sample['class_idx'],
            'cam_K': torch.from_numpy(sample['cam_K']).float(),
            'bbox': torch.tensor(sample['bbox'], dtype=torch.float32)
        }