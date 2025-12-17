"""
Dataset per PointNet-based 6D Pose Estimation.
Genera point clouds da depth + RGB images.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
import yaml

from utils.linemod_config import get_linemod_config


class LineModPointCloudDataset(Dataset):
    """
    Dataset che genera point clouds per il training con PointNet.
    
    Per ogni sample, crea una point cloud dall'oggetto cropando depth + RGB.
    """
    
    def __init__(self, root_dir, split='train', num_points=1024, use_rgb=True):
        """
        Args:
            root_dir: Path al dataset Linemod
            split: 'train' o 'test'
            num_points: Numero di punti da campionare per point cloud
            use_rgb: Se True, include RGB nella point cloud (6 canali)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.use_rgb = use_rgb
        self.config = get_linemod_config(root_dir)
        
        # Carica lista di samples
        self.samples = self._load_samples()
        
        print(f"📊 Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self):
        """Carica la lista di tutti i sample dal dataset"""
        samples = []
        
        # Itera su tutti gli oggetti (01-15)
        for obj_id in range(1, 16):
            obj_dir = self.root_dir / "data" / f"{obj_id:02d}"
            
            # Leggi il file split corrispondente
            split_file = obj_dir / f"{self.split}.txt"
            if not split_file.exists():
                continue
            
            with open(split_file, 'r') as f:
                img_ids = [int(line.strip()) for line in f.readlines()]
            
            # Aggiungi samples
            for img_id in img_ids:
                samples.append({
                    'object_id': obj_id,
                    'img_id': img_id
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_depth(self, obj_id, img_id):
        """Carica depth image in mm"""
        depth_path = self.root_dir / "data" / f"{obj_id:02d}" / "depth" / f"{img_id:04d}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)  # 16-bit
        return depth.astype(np.float32)
    
    def _load_rgb(self, obj_id, img_id):
        """Carica RGB image"""
        rgb_path = self.root_dir / "data" / f"{obj_id:02d}" / "rgb" / f"{img_id:04d}.png"
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb
    
    def _load_mask(self, obj_id, img_id):
        """Carica object mask"""
        mask_path = self.root_dir / "data" / f"{obj_id:02d}" / "mask" / f"{img_id:04d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask > 0
    
    def _load_gt_pose(self, obj_id, img_id):
        """Carica ground truth pose da cached linemod_config"""
        # Usa linemod_config che cacha automaticamente gt.yml
        return self.config.get_gt_pose(obj_id, img_id)
    
    def _get_camera_intrinsics(self, obj_id, img_id):
        """Get camera intrinsics from cached linemod_config"""
        # Usa linemod_config che cacha automaticamente info.yml
        return self.config.get_camera_intrinsics(obj_id, img_id)
    
    def _crop_with_bbox(self, image, bbox):
        """
        Croppa un'immagine usando bbox [x, y, w, h].
        
        Args:
            image: (H, W) o (H, W, C) immagine
            bbox: [x, y, w, h] bounding box
        
        Returns:
            cropped: immagine croppata
        """
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Assicurati che la bbox sia dentro l'immagine
        H, W = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        
        return image[y1:y2, x1:x2]
    
    def _depth_to_point_cloud(self, depth, cam_K, mask=None, rgb=None):
        """
        Converte depth image in point cloud.
        
        Args:
            depth: (H, W) depth in mm
            cam_K: (3, 3) camera matrix
            mask: (H, W) binary mask (opzionale)
            rgb: (H, W, 3) RGB image (opzionale)
        
        Returns:
            points: (N, 3) o (N, 6) point cloud
        """
        H, W = depth.shape
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        
        # Mesh grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Maschera punti validi
        valid_mask = depth > 0
        if mask is not None:
            valid_mask = valid_mask & mask
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]  # in mm
        
        # Back-projection
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid
        
        # Converti in metri
        points_xyz = np.stack([x/1000.0, y/1000.0, z/1000.0], axis=1)
        
        # Aggiungi RGB se disponibile
        if rgb is not None and self.use_rgb:
            rgb_valid = rgb[valid_mask]
            rgb_normalized = rgb_valid.astype(np.float32) / 255.0
            points = np.concatenate([points_xyz, rgb_normalized], axis=1)
        else:
            points = points_xyz
        
        return points
    
    def _sample_points(self, points):
        """
        Campiona un numero fisso di punti dalla point cloud.
        Usa torch.randperm per speed-up ~1.5x vs np.random.choice
        """
        n_points = len(points)
        
        if n_points >= self.num_points:
            # Random sampling con torch (più veloce)
            indices = torch.randperm(n_points)[:self.num_points].numpy()
            sampled = points[indices]
        else:
            # Pad con punti duplicati se non ci sono abbastanza punti
            indices = torch.randint(0, n_points, (self.num_points,)).numpy()
            sampled = points[indices]
        
        return sampled
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        obj_id = sample_info['object_id']
        img_id = sample_info['img_id']
        
        # 1. Carica dati
        depth = self._load_depth(obj_id, img_id)
        cam_K = self._get_camera_intrinsics(obj_id, img_id)  # Cached!
        R_gt, t_gt, bbox = self._load_gt_pose(obj_id, img_id)
        
        rgb = None
        if self.use_rgb:
            rgb = self._load_rgb(obj_id, img_id)
        
        # 2. Croppa depth e RGB usando bbox
        depth_crop = self._crop_with_bbox(depth, bbox)
        rgb_crop = self._crop_with_bbox(rgb, bbox) if rgb is not None else None
        
        # 3. Point cloud in coordinate LOCALI (relative al crop)
        # Aggiusta cam_K per il crop
        x_bbox, y_bbox, w_bbox, h_bbox = bbox
        cam_K_crop = cam_K.copy()
        cam_K_crop[0, 2] = cam_K[0, 2] - x_bbox  # cx aggiustato
        cam_K_crop[1, 2] = cam_K[1, 2] - y_bbox  # cy aggiustato
        
        # 4. Genera point cloud LOCALE
        points = self._depth_to_point_cloud(depth_crop, cam_K_crop, mask=None, rgb=rgb_crop)
        
        # 5. Campiona numero fisso di punti
        points = self._sample_points(points)
        
        # 6. Bbox info normalizzato (come nel dataset RGB)
        H, W = depth.shape
        cx = x_bbox + w_bbox / 2.0
        cy = y_bbox + h_bbox / 2.0
        bbox_info = np.array([
            cx / float(W),
            cy / float(H),
            w_bbox / float(W),
            h_bbox / float(H)
        ], dtype=np.float32)
        
        # 7. Converti rotation matrix in quaternion
        from src.pose_rgb.pose_utils import convert_rotation_to_quaternion
        quat_gt = convert_rotation_to_quaternion(torch.from_numpy(R_gt).float())
        
        return {
            'point_cloud': torch.from_numpy(points).float(),  # (num_points, 3 o 6) LOCALE
            'bbox_info': torch.from_numpy(bbox_info).float(),  # (4,) [cx%, cy%, w%, h%]
            'rotation': quat_gt,  # (4,) quaternion GLOBALE
            'translation': torch.from_numpy(t_gt).float(),  # (3,) in metri GLOBALE
            'object_id': obj_id,
            'img_id': img_id,
            'cam_K': torch.from_numpy(cam_K).float(),
            'bbox': torch.from_numpy(bbox).float()  # (4,) [x, y, w, h]
        }
