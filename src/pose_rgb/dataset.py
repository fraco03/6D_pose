import os
import cv2
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .pose_utils import convert_rotation_to_quaternion

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
        input_standard_dimensions: Optional[Tuple[int, int]] = (640, 480)
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
        
        self.object_ids = object_ids if object_ids is not None else self.VALID_OBJECTS

        self.id_to_class = {obj_id: self.CLASS_NAMES[i] for i, obj_id in enumerate(self.VALID_OBJECTS)}
        self.input_standard_dimensions = input_standard_dimensions
        self.samples = self._build_index()

        print(f" Loaded LineModPoseDataset")
        print(f"   Split: {self.split}")
        print(f"   Dir : {self.object_ids}")
        print(f"   Total samples: {len(self.samples)}")

    def _build_index(self) -> List[Dict]:
        samples = []

        for obj_id in self.object_ids:
            obj_folder = f"{obj_id:02d}"
            obj_path = self.data_dir / obj_folder

            split_file = obj_path / f"{self.split}.txt"

            if not split_file.exists():
                print(f"Warning: Split file {split_file} does not exist.")
                continue

            with open(split_file, 'r') as f:
                img_ids = [int(line.strip()) for line in f.readlines()]  # Image IDs for this split

            # Ground truth information (including poses)
            gt_file = obj_path / "gt.yml"
            with open(gt_file, 'r') as f:
                gt_data = yaml.safe_load(f)

            # Camera intrinsics file
            info_file = obj_path / "info.yml"
            with open(info_file, 'r') as f:
                info_data = yaml.safe_load(f)

            # Load samples
            for img_id in img_ids:
                img_id = int(img_id)
                if img_id not in gt_data:
                    continue  # Skip if no GT data for this image

                img_path = obj_path / 'rgb' / f"{img_id:04d}.png"

                if not img_path.exists():
                    continue  # Skip if image does not exist

                annotations = gt_data[img_id]
                for ann in annotations:
                    
                    actula_obj_id = int(ann['obj_id'])

                    rotation_matrix = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    translation_vector = np.array(ann['cam_t_m2c'])
                    quaternion_rotation = convert_rotation_to_quaternion(rotation_matrix)

                    x, y, w, h = map(int, ann['obj_bb'])
                    bbox = [x, y, w, h]

                    # If standard dimensions are provided, validate and clamp bbox now
                    if self.input_standard_dimensions is not None:
                        image_w, image_h = self.input_standard_dimensions

                        # Skip invalid size
                        if w <= 0 or h <= 0:
                            continue
                        
                        x0, y0 = x, y
                        x1, y1 = x + w, y + h
                        
                        # invalid bb 
                        if x1 <= x0 or y1 <= y0:
                            continue
                        
                        # Require bbox fully inside image bounds (inclusive on edges)
                        if not (0 <= x0 and 0 <= y0 and x1 <= image_w and y1 <= image_h):
                            continue

                        

                    sample = {
                        'object_id': actula_obj_id,
                        'class_idx': self.id_to_class[actula_obj_id],
                        'img_id': img_id,
                        'img_path': obj_path / 'rgb' / f"{img_id:04d}.png",
                        'rotation': quaternion_rotation,
                        'translation': translation_vector,
                        'bbox': ann['obj_bb'],  # [x, y, w, h]
                        'cam_K': np.array(info_data[img_id]['cam_K']).reshape(3, 3)
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

            # Crop using validated bounding box (already checked in _build_index)
            x, y, w, h = map(int, sample['bbox'])
            cropped_img = img[y:y + h, x:x + w]

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

        return {
            'image': crop_tensor,                                               # (3, 224, 224)
            'rotation': torch.from_numpy(sample['rotation']).float(),           # (4,) [w,x,y,z]
            'translation': torch.from_numpy(sample['translation']).float(),     # (3,)
            'object_id': sample['object_id'],                                   # int
            'class_idx': sample['class_idx'],                                   # int
            'cam_K': torch.from_numpy(sample['cam_K']).float(),                 # (3, 3)
            'img_id': sample['img_id']                                          # int
        }

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from class index"""
        return self.id_to_class.get(class_idx, "Unknown")



            


        