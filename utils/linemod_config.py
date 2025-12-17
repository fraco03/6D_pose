import numpy as np
import yaml
from pathlib import Path
from plyfile import PlyData
from typing import Dict, Optional


class LineModConfig:
    """
    Centralized configuration for LineMod dataset.
    Loads models_info, 3D models and provides centralized access.
    """
    
    _instance = None  # Singleton
    
    def __new__(cls, dataset_root: Optional[str] = None):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, dataset_root: Optional[str] = None):
        """
        Args:
            dataset_root: Path to Linemod_preprocessed folder
        """
        # Avoid re-initialization if already done
        if self._initialized:
            if dataset_root and dataset_root != str(self.dataset_root):
                print(f"⚠️  Warning: Dataset root already set to {self.dataset_root}")
            return
        
        if dataset_root is None:
            raise ValueError("Please specify dataset_root!")
        
        self.dataset_root = Path(dataset_root).resolve()
        self.data_dir = self.dataset_root / "data"
        self.models_dir = self.dataset_root / "models"
        
        # Check existence
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models dir not found: {self.models_dir}")
        
        # Load models_info.yml
        self.models_info = self._load_models_info()
        
        # Cache 3d models (on-demand)
        self._models_3d_cache: Dict[int, np.ndarray] = {}
        
        # Cache info.yml per object (on-demand) - contiene cam_K per ogni img_id
        self._info_cache: Dict[int, dict] = {}
        
        # Cache gt.yml per object (on-demand) - contiene pose GT per ogni img_id
        self._gt_cache: Dict[int, dict] = {}
        
        self._initialized = True
        print(f"✅ LineModConfig initialized: {self.dataset_root}")
    
    def _load_models_info(self) -> dict:
        """Load models_info.yml"""
        info_path = self.models_dir / "models_info.yml"
        if not info_path.exists():
            raise FileNotFoundError(f"models_info.yml not found: {info_path}")
        
        with open(info_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_3d(self, object_id: int, unit: str = 'mm') -> np.ndarray:
        """
        Get 3D model vertices for given object ID.
        
        Args:
            object_id: Object ID (1-15)
            unit: 'mm' (millimeters) or 'm' (meters)
        
        Returns:
            vertices: (N, 3) array of 3D points
        """
        # Check cache
        if object_id in self._models_3d_cache:
            vertices = self._models_3d_cache[object_id]
        else:
            # Load from .ply file
            ply_path = self.models_dir / f"obj_{object_id:02d}.ply"
            if not ply_path.exists():
                raise FileNotFoundError(f"Model file not found: {ply_path}")
            
            ply_data = PlyData.read(str(ply_path))
            vertices = np.vstack([
                ply_data['vertex']['x'],
                ply_data['vertex']['y'],
                ply_data['vertex']['z']
            ]).T  # (N, 3) in mm
            
            # Cache
            self._models_3d_cache[object_id] = vertices
        
        # Convert units if necessary
        if unit == 'm':
            return vertices / 1000.0
        elif unit == 'mm':
            return vertices
        else:
            raise ValueError(f"Unit must be 'mm' or 'm', not '{unit}'")
    
    def get_3d_bbox(self, object_id: int, unit: str = 'mm') -> np.ndarray:
        """
        Get the 8 vertices of the 3D bounding box
        
        Args:
            object_id: Object ID (1-15)
            unit: 'mm' or 'm'
        
        Returns:
            bbox_corners: (8, 3) array
        """
        info = self.models_info[object_id]
        
        min_x, min_y, min_z = info['min_x'], info['min_y'], info['min_z']
        max_x = min_x + info['size_x']
        max_y = min_y + info['size_y']
        max_z = min_z + info['size_z']
        
        bbox = np.array([
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ])
        
        if unit == 'm':
            return bbox / 1000.0
        return bbox
    
    def get_model_center(self, object_id: int, unit: str = 'mm') -> np.ndarray:
        """
        Get the 3D model center (centroid of vertices)
        
        Args:
            object_id: Object ID
            unit: 'mm' or 'm'
        
        Returns:
            center: (3,) array
        """
        vertices = self.get_model_3d(object_id, unit='mm')
        center = vertices.mean(axis=0)
        
        if unit == 'm':
            return center / 1000.0
        return center
    
    def get_model_info(self, object_id: int) -> dict:
        """Get model info from models_info.yml"""
        return self.models_info[object_id]
    
    def get_camera_intrinsics(self, object_id: int, img_id: int) -> np.ndarray:
        """
        Get camera intrinsics matrix for specific object and image.
        
        Args:
            object_id: Object ID (1-15)
            img_id: Image ID
        
        Returns:
            cam_K: (3, 3) camera intrinsics matrix
        """
        # Load info.yml for this object if not cached
        if object_id not in self._info_cache:
            info_file = self.data_dir / f"{object_id:02d}" / "info.yml"
            if not info_file.exists():
                raise FileNotFoundError(f"info.yml not found: {info_file}")
            
            with open(info_file, 'r') as f:
                self._info_cache[object_id] = yaml.safe_load(f)
        
        # Get cam_K for this img_id
        cam_K = np.array(self._info_cache[object_id][img_id]['cam_K']).reshape(3, 3)
        return cam_K
    
    def get_gt_pose(self, object_id: int, img_id: int):
        """
        Get ground truth pose for specific object and image.
        
        Args:
            object_id: Object ID (1-15)
            img_id: Image ID
        
        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector in meters
            bbox: (4,) bounding box [x, y, w, h]
        """
        # Load gt.yml for this object if not cached
        if object_id not in self._gt_cache:
            gt_file = self.data_dir / f"{object_id:02d}" / "gt.yml"
            if not gt_file.exists():
                raise FileNotFoundError(f"gt.yml not found: {gt_file}")
            
            with open(gt_file, 'r') as f:
                self._gt_cache[object_id] = yaml.safe_load(f)
        
        # Get pose for this img_id
        pose_data = self._gt_cache[object_id][img_id][0]  # First object
        
        # Rotation matrix (3x3)
        R = np.array(pose_data['cam_R_m2c']).reshape(3, 3)
        
        # Translation vector (mm -> meters)
        t = np.array(pose_data['cam_t_m2c']) / 1000.0
        
        # Bbox [x, y, w, h]
        bbox = np.array(pose_data['obj_bb'])
        
        return R, t, bbox
    
    def print_info(self):
        """Print dataset information"""
        print("=" * 60)
        print("📦 LineMod Dataset Configuration")
        print("=" * 60)
        print(f"Root:           {self.dataset_root}")
        print(f"Data dir:       {self.data_dir}")
        print(f"Models dir:     {self.models_dir}")
        print(f"Objects loaded: {len(self.models_info)}")
        print(f"3D models cached: {len(self._models_3d_cache)}")
        print("=" * 60)
        
        # Info for each object
        print("\n📋 Object Info:")
        for obj_id, info in self.models_info.items():
            diameter = info['diameter']
            print(f"  {obj_id:02d}: diameter={diameter:.1f}mm")

    def update_dataset_root(self, new_root: str):
        """
        Update dataset root path.
        
        Args:
            new_root: New dataset root path
        """
        self.dataset_root = Path(new_root)
        self.data_dir = self.dataset_root / "data"
        self.models_dir = self.dataset_root / "models"
        print(f"✅ Dataset root updated to: {self.dataset_root}")

    def get_dataset_root(self) -> Path:
        """Get current dataset root path"""
        return self.dataset_root


# Global singleton instance
_config_instance = None

def get_linemod_config(dataset_root: Optional[str] = None) -> LineModConfig:
    """
    Get the singleton instance of LineModConfig
    
    Args:
        dataset_root: Path to dataset (only needed on first call)
    
    Returns:
        LineModConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = LineModConfig(dataset_root)
    return _config_instance