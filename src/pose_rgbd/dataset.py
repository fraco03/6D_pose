import os
import cv2
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.pose_rgb.pose_utils import convert_rotation_to_quaternion
from utils.linemod_config import get_linemod_config

class LineModPoseDepthDataset(Dataset):
    """
    LineMod Dataset for 6D Pose Estimation with Depth Information

    Returns cropped object images, depth maps, and corresponding poses.
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
        input_standard_dimensions: Tuple[int, int] = (640, 480)
    ):
        """
        Args:
            root_dir (str): Path to the LineMod preprocessed dataset.
            split (str): 'train' or 'test' split.
            object_ids (List[int], optional): List of object IDs to include. If None, include all.
            image_size (Tuple[int, int]): Size to which images are resized.
            transform: Optional transform to be applied on a sample.
            normalize (bool): Whether to apply ImageNet normalization.
        """
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / 'data'
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize

        self.object_ids = object_ids if object_ids is not None else self.VALID_OBJECTS
        self.id_to_class = {obj_id: self.CLASS_NAMES[i] for i, obj_id in enumerate(self.VALID_OBJECTS)}
        self.input_standard_dimensions = input_standard_dimensions
        
        # Get cached config instance
        self.config = get_linemod_config(str(self.root_dir))
        
        self.samples = self._build_index()

        print(f" Loaded LineModPoseDepthDataset")
        print(f"   Split: {self.split}")
        print(f"   Objects: {self.object_ids}")
        print(f"   Total samples: {len(self.samples)}")

    def _build_index(self) -> List[Dict]:
        samples = []

        for obj_id in self.object_ids:
            # Load split file using cached config
            img_ids = self.config.get_split_file(obj_id, self.split)
            if not img_ids:
                continue

            # Load GT data using cached config
            try:
                gt_data = self.config.get_gt_data(obj_id)
            except FileNotFoundError:
                print(f"Warning: GT file not found for object {obj_id}")
                continue

            # Load camera info using cached config
            try:
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                print(f"Warning: Camera info file not found for object {obj_id}")
                continue

            obj_folder = f"{obj_id:02d}"
            obj_path = self.data_dir / obj_folder

            # Load samples
            for img_id in img_ids:
                img_id = int(img_id)
                if img_id not in gt_data:
                    continue  # Skip if no GT data for this image

                img_path = obj_path / 'rgb' / f"{img_id:04d}.png"
                depth_path = obj_path / 'depth' / f"{img_id:04d}.png"

                if not img_path.exists() or not depth_path.exists():
                    continue  # Skip if image or depth does not exist

                annotations = gt_data[img_id]
                for ann in annotations:
                    actula_obj_id = int(ann['obj_id'])

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
                    cx, cy = x + w // 2, y + h // 2
                    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if depth_map is None:
                        print(f"Warning: Could not load depth map from {depth_path}. Skipping sample.")
                        continue

                    depth_center = depth_map[cy, cx] / 1000.0  # Convert mm to meters

                    # Calculate 3D coordinates using inverse pinhole
                    cam_K = np.array(info_data[img_id]['cam_K']).reshape(3, 3)
                    fx, fy = cam_K[0, 0], cam_K[1, 1]
                    cx_k, cy_k = cam_K[0, 2], cam_K[1, 2]

                    X = (cx - cx_k) * depth_center / fx
                    Y = (cy - cy_k) * depth_center / fy
                    Z = depth_center

                    sample = {
                        'object_id': actula_obj_id,
                        'class_idx': self.id_to_class[actula_obj_id],
                        'img_id': img_id,
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
        cropped_img = img[y:y + h, x:x + w]
        cropped_depth = depth_map[y:y + h, x:x + w]

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