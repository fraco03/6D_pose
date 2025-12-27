import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import plyfile

# Import your custom modules (ensure these are accessible)
from src.pose_rgbd.model import RGBDRotationModel 
from utils.load_data import load_model
from utils.projection_utils import project_points_to_2dimage
from src.inference.inference_visualize import compute_and_draw_prediction

from src.pose_rgb.model import ResNetRotation, TranslationNet

class PoseInferencePipeline:
    def __init__(
        self, 
        yolo_model,
        pose_model,
        models_path: str, 
        yolo_to_linemod_map: dict = None,
        device: str = 'cuda',
        conf_threshold: float = 0.5,
        prepare_sample_func = None,
        inference_func = None
    ):
        """
        Initializes the pipeline by loading models and caching 3D info.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold

        # self.object_names = [
        #     'Ape', 'Benchvise', 'Camera', 'Can', 'Cat',
        #     'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher',
        #     'Iron', 'Lamp', 'Phone'
        # ]

        self.object_names = {
            1: 'Ape', 2: 'Benchvise', 4: 'Camera', 5: 'Can', 6: 'Cat',
            8: 'Driller', 9: 'Duck', 10: 'Eggbox', 11: 'Glue', 12: 'Holepuncher',
            13: 'Iron', 14: 'Lamp', 15: 'Phone'
        }

        # Default mapping if none provided
        if yolo_to_linemod_map is None:
            LINEMOD_TO_YOLO_ID = {
                '1': 0,   '2': 1,   '4': 2,   '5': 3,   '6': 4,
                '8': 5,   '9': 6,   '10': 7,  '11': 8,  '12': 9,
                '13': 10, '14': 11, '15': 12
            }
            # Reverse mapping: YOLO ID -> Linemod ID
            self.yolo_to_linemod_map = {v: int(k) for k, v in LINEMOD_TO_YOLO_ID.items()}
        else:
            self.yolo_to_linemod_map = yolo_to_linemod_map

        # 1. Load Models
        self.yolo_model = yolo_model # Assumed already to(device) if needed, or handled internally
        self.pose_model = pose_model.to(self.device)
        self.pose_model.eval()

        # Functions for specific model logic
        self.prepare_sample_func = prepare_sample_func
        self.inference_func = inference_func

        # 2. Cache 3D BBoxes
        print("📦 Caching 3D model bounding boxes...")
        self.models_3d_bboxes = self._read_models_bounding_boxes(models_path / "models_info.yml", unit='m')

        # 3. Cache 3D Model Points (Optional, for detailed visualization)
        print("📦 Caching 3D model points...")
        models_dir = models_path
        self.models_3d_points = self._read_models_points(str(models_dir))

        print("✅ Pipeline initialization complete.")

    def _read_models_bounding_boxes(self, models_info_path: str, unit='m') -> dict:
        """
        Reads models_info.yml and pre-calculates the 8 corners of the 3D bbox for each object.
        """
        models_info_path = Path(models_info_path)
        if not models_info_path.exists():
            raise FileNotFoundError(f"models_info.yml not found: {models_info_path}")

        with open(models_info_path, 'r') as file:
            models_info = yaml.safe_load(file)

        model_bb = {}
        for model_id, model_data in models_info.items():
            min_x, min_y, min_z = model_data['min_x'], model_data['min_y'], model_data['min_z']
            size_x, size_y, size_z = model_data['size_x'], model_data['size_y'], model_data['size_z']
            
            max_x, max_y, max_z = min_x + size_x, min_y + size_y, min_z + size_z

            bbox = np.array([
                [min_x, min_y, min_z], [max_x, min_y, min_z],
                [max_x, max_y, min_z], [min_x, max_y, min_z],
                [min_x, min_y, max_z], [max_x, min_y, max_z],
                [max_x, max_y, max_z], [min_x, max_y, max_z],
            ])

            if unit == 'm':
                bbox = bbox / 1000.0
            
            model_bb[int(model_id)] = bbox
        
        return model_bb

    def _read_models_points(self, models_path: str) -> dict:
        models_points = {}
        # Simple glob, might need adjustment if filenames vary
        for model_file in Path(models_path).glob("obj_*.ply"): 
            # Extract ID from filename "obj_01.ply" -> 1
            try:
                model_id = int(model_file.stem.split('_')[1])
                models_points[model_id] = self._read_model_points(str(model_file))
            except (IndexError, ValueError):
                continue
        return models_points

    def _read_model_points(self, model_path: str) -> np.ndarray:
        # Requires 'plyfile' package
        with open(model_path, 'rb') as f:
            plydata = plyfile.PlyData.read(f)
            vertices = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
            # Convert mm to m if your pipeline uses meters (Linemod standard is mm in ply usually)
            vertices = vertices / 1000.0 
        return vertices

    def run_on_folder(self, sample_folder: str, cam_K: np.ndarray) -> list:
        """
        Runs inference on a specific folder containing sample data.
        """
        sample_path = Path(sample_folder)
        rgb_path = sample_path / "rgb.png"
        depth_path = sample_path / "depth.png"

        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB not found in: {sample_path}")

        # Load Images
        rgb_image = cv2.imread(str(rgb_path)) # BGR
        depth_image = None
        if depth_path.exists():
            depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        return self._run_inference(rgb_image, depth_image, cam_K)

    def _run_inference(self, rgb_image, depth_image, cam_K):
        """
        Core inference logic: YOLO -> Prepare Input -> Model Inference -> Projection.
        """
        if self.prepare_sample_func is None or self.inference_func is None:
            raise ValueError("prepare_sample_func and inference_func must be defined.")

        # Ensure RGB input for model (OpenCV loads BGR)
        # Note: Your prepare_sample might expect BGR or RGB, checking below it seems you convert inside?
        # Let's assume input to this func is standard OpenCV BGR.

        vis_img = rgb_image.copy()
        
        # 1. YOLO Inference
        results = self.yolo_model.predict(source=rgb_image, verbose=False, conf=self.conf_threshold)
        
        detections = []

        for result in results:
            for box in result.boxes:
                # YOLO Data
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Map YOLO ID -> Linemod ID
                obj_id = self.yolo_to_linemod_map.get(cls_id)
                if obj_id is None: continue 

                bbox = [x1, y1, x2, y2]

                # 2. Prepare Input Sample (Model specific)
                # prepare_sample_func should return a dict or whatever inference_func expects
                sample_input = self.prepare_sample_func(rgb_image, depth_image, bbox, self.device)
                
                if sample_input is None: continue

                # 3. Model Inference
                # Should return rotation (quaternion/matrix) and translation vector
                pred_rot, pred_trans = self.inference_func(sample_input, self.pose_model, cam_K)
                
                if pred_trans is None: continue

                # 4. 3D Projection (Visualization Data)
                bbox_3d_model = self.models_3d_bboxes.get(obj_id)
                bbox_3d_proj = None
                
                if bbox_3d_model is not None:
                    # Project using the generic utility
                    bbox_3d_proj = project_points_to_2dimage(
                        bbox_3d_model, cam_K, pred_rot, pred_trans
                    )

                    vis_img = compute_and_draw_prediction(
                        vis_img,       # Passiamo l'immagine inizializzata sopra
                        bbox_3d_model, 
                        cam_K,
                        pred_rot,
                        pred_trans,
                        conf_score=conf,
                        label=f"{self.object_names.get(obj_id, 'Unknown')}"
                    )

                    

                detections.append({
                    'class_id': obj_id,
                    'yolo_class': cls_id,
                    'conf': conf,
                    'bbox_2d': bbox,
                    'rotation': pred_rot,
                    'translation': pred_trans,
                    'bbox_3d_proj': bbox_3d_proj
                })
        
        return (detections, vis_img)


class RGBDPoseInferencePipeline:
    def __init__(
        self,
        yolo_path,
        pose_model_path,
        models_path: str,
        yolo_to_linemod_map: dict = None,
        device: str = 'cuda',
        conf_threshold: float = 0.5
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize YOLO
        print("📦 Loading YOLO model...")
        yolo_model = YOLO(yolo_path)

        # 2. Initialize Pose Model
        print("📦 Loading RGB-D Pose model...")
        pose_model = load_model(
            checkpoint_location=str(pose_model_path), # ensure string path
            device=self.device,
            model_class=RGBDRotationModel,
            model_key='model_state_dict',
            pretrained=False # Usually False for inference loading
        )

        # 3. Define the specific prepare function for RGB-D
        def prepare_input_rgbd(rgb_full_bgr, depth_full, bbox, device):
            x1, y1, x2, y2 = map(int, bbox)
            h_img, w_img = rgb_full_bgr.shape[:2]
            
            # Clip bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            rgb_crop = rgb_full_bgr[y1:y2, x1:x2]
            depth_crop = depth_full[y1:y2, x1:x2]
            
            if rgb_crop.size == 0 or depth_crop.size == 0: 
                return None
            
            # Convert BGR to RGB for the model
            rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB)

            # Resize standard ResNet
            rgb_crop = cv2.resize(rgb_crop, (224, 224))
            depth_crop = cv2.resize(depth_crop, (224, 224), interpolation=cv2.INTER_NEAREST)
            
            # Normalize RGB
            rgb_norm = rgb_crop.astype(np.float32) / 255.0
            rgb_norm = (rgb_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Normalize Depth (2m max depth assumption)
            depth_norm = depth_crop.astype(np.float32)
            depth_norm = np.clip(depth_norm / 2000.0, 0.0, 1.0)
            
            rgb_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)
            depth_tensor = torch.from_numpy(depth_norm).float().unsqueeze(0).unsqueeze(0).to(device)

            # Return a dict containing all necessary data for inference
            return {
                'rgb': rgb_tensor,
                'depth_tensor': depth_tensor,
                'depth_full': depth_full, # Needed for geometric translation
                'bbox': bbox
            }

        # 4. Define the specific inference function
        def inference_func(sample_input: dict, model, cam_K):
            rgb_in = sample_input['rgb']
            depth_in = sample_input['depth_tensor']
            depth_full = sample_input['depth_full']
            bbox = sample_input['bbox']

            # Rotation Inference
            with torch.no_grad():
                pred_quat = model(rgb_in, depth_in)
                pred_quat = pred_quat.cpu().numpy().flatten()
                pred_quat = pred_quat / np.linalg.norm(pred_quat)

            # Translation Inference (Geometric)
            x1, y1, x2, y2 = map(int, bbox)
            cx_box = (x1 + x2) // 2
            cy_box = (y1 + y2) // 2
            
            # Handle depth reading
            z_raw = depth_full[cy_box, cx_box]
            if z_raw == 0:
                crop = depth_full[y1:y2, x1:x2]
                valid_pixels = crop[crop > 0]
                if valid_pixels.size > 0:
                    z_raw = np.median(valid_pixels)
                else:
                    return pred_quat, None # Translation failed
            
            Z = z_raw / 1000.0  # mm -> Meters
            
            fx, fy = cam_K[0, 0], cam_K[1, 1]
            cx_k, cy_k = cam_K[0, 2], cam_K[1, 2]
            
            X = (cx_box - cx_k) * Z / fx
            Y = (cy_box - cy_k) * Z / fy
            
            pred_trans = np.array([X, Y, Z], dtype=np.float32)
            
            return pred_quat, pred_trans

        # 5. Instantiate the Base Pipeline
        self.pipeline = PoseInferencePipeline(
            yolo_model=yolo_model,
            pose_model=pose_model,
            models_path=models_path,
            yolo_to_linemod_map=yolo_to_linemod_map,
            device=self.device.type,
            conf_threshold=conf_threshold,
            prepare_sample_func=prepare_input_rgbd,
            inference_func=inference_func
        )

    def run(self, sample_folder, cam_K: np.ndarray):
        """Wrapper to call the underlying pipeline's run method."""

        return self.pipeline.run_on_folder(sample_folder, cam_K)
    
import torch.nn as nn

class RGBPoseInferencePipeline:
    def __init__(
        self,
        yolo_path,
        rot_model_path,
        trans_model_path,
        models_path: str,
        yolo_to_linemod_map: dict = None,
        device: str = 'cuda',
        conf_threshold: float = 0.5,
        std_dims: tuple = (640, 480) # Dimensioni usate durante il training per normalizzare bbox_info
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize YOLO
        print("📦 Loading YOLO model...")
        yolo_model = YOLO(yolo_path)

        # 2. Initialize Pose Models
        print("📦 Loading RGB Pose models (Rotation & Translation)...")
        
        rot_model = load_model(
            checkpoint_location=str(rot_model_path),
            device=self.device,
            model_class=ResNetRotation,
            model_key='model_state_dict'
        )

        trans_model = load_model(
            checkpoint_location=str(trans_model_path),
            device=self.device,
            model_class=TranslationNet,
            model_key='model_state_dict'
        )

        # AVVOLGIMENTO FONDAMENTALE:
        # Usiamo ModuleList affinché la classe base possa fare self.pose_model.to(device) e .eval()
        # senza rompere il fatto che sono due modelli distinti.
        pose_model_wrapper = nn.ModuleList([rot_model, trans_model])

        # ---------------------------------------------------------
        # Funzione di Preparazione Input
        # ---------------------------------------------------------
        def prepare_input_rgb(rgb_full_bgr, depth_full, bbox, device):
            # Nota: depth_full viene ignorata (o usata come fallback se volessi),
            # ma la firma deve accettarla per compatibilità con la pipeline generica.
            
            x1, y1, x2, y2 = map(int, bbox)
            h_img, w_img = rgb_full_bgr.shape[:2]

            # 1. Boundary Checks & Robustness
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None

            rgb_crop = rgb_full_bgr[y1:y2, x1:x2]
            if rgb_crop.size == 0:
                return None

            # 2. Image Preprocessing (Standard ResNet/ImageNet)
            rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB)
            rgb_crop = cv2.resize(rgb_crop, (224, 224)) # Dimensione input modello
            
            rgb_norm = rgb_crop.astype(np.float32) / 255.0
            rgb_norm = (rgb_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # (1, 3, 224, 224)
            img_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)

            # 3. Prepare BBox Info for TranslationNet
            # Calcolo le feature geometriche normalizzate rispetto alle dimensioni standard di training (640x480)
            
            # Width e Height attuali del box
            w_box = x2 - x1
            h_box = y2 - y1
            
            # Centro attuale del box (in pixel immagine originale)
            cx_real = x1 + w_box / 2.0
            cy_real = y1 + h_box / 2.0
            
            # Normalizzazione (Training assumption: 640x480)
            std_w, std_h = std_dims 
            
            bbox_info = torch.tensor([
                cx_real / float(std_w),
                cy_real / float(std_h),
                w_box / float(std_w),
                h_box / float(std_h)
            ], dtype=torch.float32).unsqueeze(0).to(device) # (1, 4)

            # Serve anche il centro reale e le dimensioni di normalizzazione per la post-processing
            bbox_center = torch.tensor([cx_real, cy_real], dtype=torch.float32).unsqueeze(0).to(device)
            
            return {
                'image': img_tensor,
                'bbox_info': bbox_info,
                'bbox_center': bbox_center,
                'std_w': std_w,
                'std_h': std_h
            }

        # ---------------------------------------------------------
        # Funzione Helper: Inverse Pinhole Projection
        # ---------------------------------------------------------
        def inverse_pinhole_projection(crop_center, deltas, z, cam_K):
            # crop_center: (Batch, 2) -> (cx_box, cy_box)
            # deltas: (Batch, 2) -> (delta_x, delta_y) in pixel
            # z: (Batch, 1) -> profondità stimata in metri
            # cam_K: (3, 3) matrice intrinseca
            
            fx = cam_K[0, 0]
            fy = cam_K[1, 1]
            cx_cam = cam_K[0, 2]
            cy_cam = cam_K[1, 2]

            # Il modello predice un delta rispetto al centro del crop
            # Pixel finale stimato = centro_bbox + delta
            u_pred = crop_center[:, 0] + deltas[:, 0]
            v_pred = crop_center[:, 1] + deltas[:, 1]

            # Back-projection: (u - cx) * Z / fx
            X = (u_pred - cx_cam) * z[:, 0] / fx
            Y = (v_pred - cy_cam) * z[:, 0] / fy
            Z = z[:, 0]

            return torch.stack([X, Y, Z], dim=1) # (Batch, 3)

        # ---------------------------------------------------------
        # Funzione di Inferenza
        # ---------------------------------------------------------
        def inference_func(sample_input: dict, models_wrapper, cam_K_np):
            # Unpack models from ModuleList
            rot_net = models_wrapper[0]
            trans_net = models_wrapper[1]
            
            # Unpack Data
            img_tensor = sample_input['image']
            bbox_info = sample_input['bbox_info']
            bbox_center = sample_input['bbox_center']
            
            # Nota: cam_K qui arriva come numpy dalla pipeline base, lo convertiamo in tensor
            cam_K_tensor = torch.from_numpy(cam_K_np).float().to(self.device)

            # 1. Rotation Inference
            with torch.no_grad():
                # Forward Rotazione
                pred_quat = rot_net(img_tensor)
                # Normalizzazione quaternione
                pred_quat = pred_quat / torch.norm(pred_quat, dim=1, keepdim=True)
                pred_quat = pred_quat.cpu().numpy().flatten() # Ritorna array (4,)

                # 2. Translation Inference
                # Forward TranslationNet
                preds_raw = trans_net(img_tensor, bbox_info) # (1, 3) -> dx, dy, log_z
                
                # Split outputs
                pred_relative_deltas = preds_raw[:, :2] # (1, 2)
                pred_log_z = preds_raw[:, 2:3]          # (1, 1)

                # Denormalizzazione Deltas
                # bbox_info contiene [cx_norm, cy_norm, w_norm, h_norm]
                # Le dimensioni reali (pixel) si ricavano rimoltiplicando per std_dims
                w_norm = bbox_info[:, 2]
                h_norm = bbox_info[:, 3]
                
                std_w = sample_input['std_w']
                std_h = sample_input['std_h']

                W_real = w_norm * std_w
                H_real = h_norm * std_h
                
                real_dims = torch.stack([W_real, H_real], dim=1) # (1, 2)
                
                # Calcolo spostamento in pixel
                pred_deltas = pred_relative_deltas * real_dims

                # Calcolo Z
                pred_z = torch.exp(pred_log_z).clamp(min=0.1, max=10.0)

                # Ricostruzione 3D
                pred_trans_tensor = inverse_pinhole_projection(
                    crop_center=bbox_center,
                    deltas=pred_deltas,
                    z=pred_z,
                    cam_K=cam_K_tensor
                )
                
                pred_trans = pred_trans_tensor.cpu().numpy().flatten() # Array (3,)

            return pred_quat, pred_trans

        # ---------------------------------------------------------
        # Istanziazione Pipeline Base
        # ---------------------------------------------------------
        self.pipeline = PoseInferencePipeline(
            yolo_model=yolo_model,
            pose_model=pose_model_wrapper, # Passiamo la ModuleList
            models_path=models_path,
            yolo_to_linemod_map=yolo_to_linemod_map,
            device=self.device.type,
            conf_threshold=conf_threshold,
            prepare_sample_func=prepare_input_rgb,
            inference_func=inference_func
        )

    def run(self, sample_folder, cam_K: np.ndarray):
        """Wrapper to call the underlying pipeline's run method."""
        return self.pipeline.run_on_folder(sample_folder, cam_K)
    

import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
from src.pose_pointnet.model import PointNetPoseModel  # Assumo questo sia il percorso del tuo modello
from utils.load_data import load_model

# Importa la classe base (assicurati che questo import corrisponda alla struttura del tuo progetto)
# from src.inference.inference_pipeline import PoseInferencePipeline 

class PointNetInferencePipeline:
    def __init__(
        self,
        yolo_path,
        pointnet_model_path,
        models_path: str,
        yolo_to_linemod_map: dict = None,
        device: str = 'cuda',
        conf_threshold: float = 0.5,
        num_points: int = 1024
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_points = num_points
        
        # 1. Initialize YOLO
        print("🔥 Loading YOLO model (Detector)...")
        yolo_model = YOLO(yolo_path)

        # 2. Initialize PointNet Pose Model
        print("🔥 Loading PointNet Pose model (Geometry)...")
        pose_model = load_model(
            checkpoint_location=str(pointnet_model_path),
            device=self.device,
            model_class=PointNetPoseModel, 
            model_key='model_state_dict'
        )
        
        # ---------------------------------------------------------
        # Funzione di Preparazione Input (con FIX dimensione canali)
        # ---------------------------------------------------------
        def prepare_input_pointnet(rgb_full, depth_full, bbox, device):
            if depth_full is None:
                return None
            
            # --- FIX CRITICO: Gestione canali depth ---
            # Se la depth ha 3 canali (H, W, 3), ne prendiamo solo uno.
            # Questo previene l'errore "too many values to unpack" e la duplicazione dei punti.
            if len(depth_full.shape) == 3:
                depth_full = depth_full[:, :, 0]
            # ------------------------------------------

            x, y, x2, y2 = map(int, bbox)
            w = x2 - x
            h = y2 - y
            
            # Boundary Checks
            img_h, img_w = depth_full.shape[:2]
            x = max(0, x); y = max(0, y)
            w = min(w, img_w - x); h = min(h, img_h - y)
            
            if w <= 0 or h <= 0: return None

            # 1. Crop Depth
            depth_crop = depth_full[y:y+h, x:x+w]
            
            # 2. Valid Mask Check
            valid_mask = depth_crop > 0
            if not np.any(valid_mask): return None
            
            return {
                'depth_crop': depth_crop,
                'bbox_coords': (x, y, w, h),
                'device': device
            }

        # ---------------------------------------------------------
        # Funzione di Inferenza
        # ---------------------------------------------------------
        def inference_func(sample_input: dict, model, cam_K):
            # 1. Retrieve Data
            depth_crop = sample_input['depth_crop']
            x, y, w, h = sample_input['bbox_coords']
            dev = sample_input['device']
            
            # 2. Back-projection
            fx, fy = cam_K[0, 0], cam_K[1, 1]
            cx, cy = cam_K[0, 2], cam_K[1, 2]
            
            # Ora .shape restituirà sempre 2 valori grazie al fix in prepare
            rows, cols = depth_crop.shape 
            c, r = np.meshgrid(np.arange(cols), np.arange(rows))
            
            # Global coordinates
            u_vals = c + x
            v_vals = r + y
            
            valid_mask = depth_crop > 0
            if not np.any(valid_mask): return None, None
            
            z_vals = depth_crop[valid_mask] / 1000.0 # mm -> Meters
            u_vals = u_vals[valid_mask]
            v_vals = v_vals[valid_mask]
            
            x_vals = (u_vals - cx) * z_vals / fx
            y_vals = (v_vals - cy) * z_vals / fy
            
            points = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)
            
            # 3. Sampling
            num_points_avail = points.shape[0]
            if num_points_avail == 0: return None, None
            
            if num_points_avail >= self.num_points:
                choice_idx = np.random.choice(num_points_avail, self.num_points, replace=False)
            else:
                choice_idx = np.random.choice(num_points_avail, self.num_points, replace=True)
            
            points = points[choice_idx, :] # (1024, 3)
            
            # 4. Centering
            centroid = np.mean(points, axis=0) # (3,)
            points_centered = points - centroid
            
            # Prepare Tensor: (Batch, Channels, Points) -> (1, 3, 1024)
            points_tensor = torch.from_numpy(points_centered).T.unsqueeze(0).float().to(dev)
            
            # 5. Forward Pass
            # model.eval() # Già impostato nella base class
            with torch.no_grad():
                pred_q, pred_t_res = model(points_tensor)
                
                # Normalize Quaternion
                pred_q = pred_q / torch.norm(pred_q, dim=1, keepdim=True)
                pred_q = pred_q.cpu().numpy().flatten()
                
                # 6. Reconstruction
                pred_t_res = pred_t_res.cpu().numpy().flatten()
                pred_trans = centroid + pred_t_res
            
            return pred_q, pred_trans

        # ---------------------------------------------------------
        # Istanziazione Pipeline Base
        # ---------------------------------------------------------
        self.pipeline = PoseInferencePipeline(
            yolo_model=yolo_model,
            pose_model=pose_model,
            models_path=models_path,
            yolo_to_linemod_map=yolo_to_linemod_map,
            device=self.device.type,
            conf_threshold=conf_threshold,
            prepare_sample_func=prepare_input_pointnet,
            inference_func=inference_func
        )

    def run(self, sample_folder, cam_K: np.ndarray):
        return self.pipeline.run_on_folder(sample_folder, cam_K)
    

import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
from src.dense_fusion.model import FusionPoseModel  # Adatta l'import alla tua struttura
from utils.load_data import load_model

class DenseFusionInferencePipeline:
    def __init__(
        self,
        yolo_path,
        densefusion_model_path,
        models_path: str,
        yolo_to_linemod_map: dict = None,
        device: str = 'cuda',
        conf_threshold: float = 0.5,
        num_points: int = 1024,
        resize_shape: tuple = (128, 128)
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_points = num_points
        self.resize_shape = resize_shape
        
        # 1. Initialize YOLO
        print("🔥 Loading YOLO model (Detector)...")
        yolo_model = YOLO(yolo_path)

        # 2. Initialize DenseFusion Model
        print("🔥 Loading DenseFusion model (RGB-D Fusion)...")
        pose_model = load_model(
            checkpoint_location=str(densefusion_model_path),
            device=self.device,
            model_class=FusionPoseModel, 
            model_key='model_state_dict'
        )
        
        # ---------------------------------------------------------
        # Funzione di Preparazione Input
        # ---------------------------------------------------------
        def prepare_input_densefusion(rgb_full, depth_full, bbox, device):
            if depth_full is None: return None
            
            # Fix canali depth
            if len(depth_full.shape) == 3:
                depth_full = depth_full[:, :, 0]

            x, y, x2, y2 = map(int, bbox)
            w_box = x2 - x
            h_box = y2 - y
            
            # Boundary Checks
            img_h, img_w = depth_full.shape[:2]
            x = max(0, x); y = max(0, y)
            w_box = min(w_box, img_w - x); h_box = min(h_box, img_h - y)
            
            if w_box <= 0 or h_box <= 0: return None

            # Crop RGB & Depth
            depth_crop = depth_full[y:y+h_box, x:x+w_box]
            rgb_crop = rgb_full[y:y+h_box, x:x+w_box] # BGR
            
            valid_mask = depth_crop > 0
            if not np.any(valid_mask): return None
            
            return {
                'rgb_crop': rgb_crop,
                'depth_crop': depth_crop,
                'bbox_coords': (x, y, w_box, h_box),
                'device': device
            }

        # ---------------------------------------------------------
        # Funzione di Inferenza
        # ---------------------------------------------------------
        def inference_func(sample_input: dict, model, cam_K):
            # 1. Retrieve Data
            rgb_crop_bgr = sample_input['rgb_crop']
            depth_crop = sample_input['depth_crop']
            x, y, w, h = sample_input['bbox_coords']
            dev = sample_input['device']
            
            rgb_crop = cv2.cvtColor(rgb_crop_bgr, cv2.COLOR_BGR2RGB)

            # 2. Back-projection
            fx, fy = cam_K[0, 0], cam_K[1, 1]
            cx, cy = cam_K[0, 2], cam_K[1, 2]
            
            rows, cols = depth_crop.shape
            c, r = np.meshgrid(np.arange(cols), np.arange(rows))
            
            u_vals = c + x
            v_vals = r + y
            
            valid_mask = depth_crop > 0
            if not np.any(valid_mask): return None, None
            
            z_vals = depth_crop[valid_mask] / 1000.0 # mm -> m
            u_vals = u_vals[valid_mask]
            v_vals = v_vals[valid_mask]
            
            x_vals = (u_vals - cx) * z_vals / fx
            y_vals = (v_vals - cy) * z_vals / fy
            
            points = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)
            
            # 3. Sampling
            num_valid = points.shape[0]
            if num_valid == 0: return None, None

            if num_valid >= self.num_points:
                choice_idx = np.random.choice(num_valid, self.num_points, replace=False)
            else:
                choice_idx = np.random.choice(num_valid, self.num_points, replace=True)
            
            points = points[choice_idx, :] # (1024, 3)

            # 4. RGB Resizing (Necessario per matchare l'input della CNN)
            target_h, target_w = self.resize_shape
            rgb_resized = cv2.resize(rgb_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # 5. Normalization & Tensor prep
            centroid = np.mean(points, axis=0)
            points_centered = points - centroid
            
            # Tensors
            # Points: (1, 3, 1024)
            points_tensor = torch.from_numpy(points_centered).T.unsqueeze(0).float().to(dev)
            
            # RGB: (1, 3, H, W) normalized 0-1
            rgb_tensor = torch.from_numpy(rgb_resized).float() / 255.0
            rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(dev)

            # 6. Forward Pass
            # model.eval() # Già settato nella base class
            with torch.no_grad():
                # --- FIX: Passiamo solo points e images, come da tuo snippet ---
                pred_q, pred_t_res = model(points_tensor, rgb_tensor)
                
                # Normalize & Reconstruct
                pred_q = pred_q / torch.norm(pred_q, dim=1, keepdim=True)
                pred_q = pred_q.cpu().numpy().flatten()
                
                pred_t_res = pred_t_res.cpu().numpy().flatten()
                pred_trans = centroid + pred_t_res
            
            return pred_q, pred_trans

        # ---------------------------------------------------------
        # Istanziazione Pipeline Base
        # ---------------------------------------------------------
        self.pipeline = PoseInferencePipeline(
            yolo_model=yolo_model,
            pose_model=pose_model,
            models_path=models_path,
            yolo_to_linemod_map=yolo_to_linemod_map,
            device=self.device.type,
            conf_threshold=conf_threshold,
            prepare_sample_func=prepare_input_densefusion,
            inference_func=inference_func
        )

    def run(self, sample_folder, cam_K: np.ndarray):
        return self.pipeline.run_on_folder(sample_folder, cam_K)