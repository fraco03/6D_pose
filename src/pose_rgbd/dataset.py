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
    Optimized: Heavy lifting moved to __getitem__.
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
        random_seed: int = 42
    ):
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

        print(f" Loaded LineModPoseDepthDataset")
        print(f"   Split: {self.split} (Ratio: {self.train_ratio})")
        print(f"   Objects: {self.object_ids}")
        print(f"   Total samples: {len(self.samples)}")

    def _build_index(self) -> List[Dict]:
        samples = []

        for obj_id in self.object_ids:
            try:
                gt_data = self.config.get_gt_data(obj_id)
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                print(f"Warning: Data files not found for object {obj_id}")
                continue

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
                # Accesso sicuro alle chiavi (int/str/padded)
                annotations = gt_data.get(img_id_int) or gt_data.get(str(img_id_int)) or gt_data.get(f"{img_id_int:04d}")
                
                if not annotations:
                    continue

                img_path = obj_path / 'rgb' / f"{img_id_int:04d}.png"
                depth_path = obj_path / 'depth' / f"{img_id_int:04d}.png"

                # Check esistenza file veloce (senza leggere)
                if not img_path.exists() or not depth_path.exists():
                    continue

                for ann in annotations:
                    actual_obj_id = int(ann['obj_id'])

                    # FILTRO CRUCIALE: Solo l'oggetto corrente
                    if actual_obj_id != obj_id:
                        continue

                    x, y, w, h = map(int, ann['obj_bb'])
                    
                    # Validazione BBox (Identica all'RGB Dataset)
                    image_w, image_h = self.input_standard_dimensions
                    if w <= 0 or h <= 0: continue
                    
                    x0, y0 = x, y
                    x1, y1 = x + w, y + h
                    if x1 <= x0 or y1 <= y0: continue
                    if not (0 <= x0 and 0 <= y0 and x1 <= image_w and y1 <= image_h):
                        continue

                    # Recupero Info Camera
                    cam_info = info_data.get(img_id_int) or info_data.get(str(img_id_int)) or info_data.get(f"{img_id_int:04d}")
                    if cam_info is None:
                        continue
                    
                    # Prepara i dati grezzi (converti qui le matrici per risparmiare tempo dopo)
                    rotation_matrix = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    translation_vector = np.array(ann['cam_t_m2c'])
                    quaternion_rotation = convert_rotation_to_quaternion(rotation_matrix)
                    cam_K = np.array(cam_info['cam_K']).reshape(3, 3)

                    # Salviamo solo i metadati, niente immagini o calcoli 3D qui!
                    sample = {
                        'object_id': actual_obj_id,
                        'class_idx': self.id_to_class[actual_obj_id],
                        'img_id': img_id_int,
                        'img_path': str(img_path),
                        'depth_path': str(depth_path),
                        'rotation': quaternion_rotation,
                        'translation': translation_vector / 1000.0, # m
                        'bbox': [x, y, w, h],
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
            # Fallback
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Load Depth Map
        depth_map = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED)
        
        # Gestione errori depth
        if depth_map is None:
            depth_map = np.zeros((self.input_standard_dimensions[1], self.input_standard_dimensions[0]), dtype=np.float32)
            Z_center = 0.0
        else:
            # 3. Calcolo Centro 3D (Spostato qui!)
            x, y, w, h = map(int, sample['bbox'])
            img_h, img_w = depth_map.shape
            
            # Centro del bbox
            cx = min(max(x + w // 2, 0), img_w - 1)
            cy = min(max(y + h // 2, 0), img_h - 1)
            
            # Valore Depth al centro (convertito in metri)
            Z_center = depth_map[cy, cx] / 1000.0

        # Back-projection (Pinhole Model)
        cam_K = sample['cam_K']
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx_k, cy_k = cam_K[0, 2], cam_K[1, 2]

        # X = (u - cx) * Z / fx
        X_center = (cx - cx_k) * Z_center / fx
        Y_center = (cy - cy_k) * Z_center / fy
        
        center_3d = np.array([X_center, Y_center, Z_center], dtype=np.float32)

        # 4. Crop & Resize Image
        x, y, w, h = map(int, sample['bbox'])
        # Safety clip again for crop
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.input_standard_dimensions[0], x + w)
        y1 = min(self.input_standard_dimensions[1], y + h)

        cropped_img = img[y0:y1, x0:x1]
        cropped_img = cv2.resize(cropped_img, self.image_size)

        # 5. Crop & Resize Depth (Channel 4)
        cropped_depth = depth_map[y0:y1, x0:x1]
        # Clip e resize depth
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
        depth_max = 2000.0 # 2m max depth assumption for normalization
        depth_tensor = torch.clamp(depth_tensor / depth_max, 0.0, 1.0).unsqueeze(0)

        return {
            'image': img_tensor,
            'depth': depth_tensor,
            'img_id': sample['img_id'],
            'img_path': sample['img_path'],
            'rotation': torch.from_numpy(sample['rotation']).float(),
            'translation': torch.from_numpy(sample['translation']).float(),
            '3D_center': torch.from_numpy(center_3d).float(), # Ecco il valore calcolato "on the fly"
            'object_id': sample['object_id'],
            'class_idx': sample['class_idx'],
            'cam_K': torch.from_numpy(sample['cam_K']).float(),
            'bbox': torch.tensor(sample['bbox'], dtype=torch.float32)
        }