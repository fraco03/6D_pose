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

import _pickle as cPickle
from PIL import Image
import torchvision.transforms as transforms

class DenseFusionLineModDataset(Dataset):
    """
    LineMod Dataset for PointNet-based 6D Pose Estimation.
    
    It converts 2D Depth maps into 3D Point Clouds via back-projection
    and samples a fixed number of points (e.g., 1024).
    """

    VALID_OBJECTS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    
    CLASS_NAMES = [
        'ape', 'benchvise', 'camera', 'can', 'cat',
        'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
        'iron', 'lamp', 'phone'
    ]

    def __init__(self, root_dir, split='train', num_points=1024, augment=False, verbose=True):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        
        # Caricamento liste file (come nel tuo codice originale)
        self.list_of_points = []
        self.list_of_labels = []
        
        # Valid objects map (preso dal tuo codice)
        self.id_to_class = {
            1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat',
            8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue',
            12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'
        }
        self.VALID_OBJECTS = list(self.id_to_class.keys())

        # Logica di caricamento (semplificata per brevità, mantieni la tua se complessa)
        split_file = os.path.join(root_dir, f"{split}.txt")
        with open(split_file, 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        for line in file_names:
            # line es: data/01/0000 (classe/id)
            # Costruiamo il path al file .dat preprocessato
            # Assumo che i tuoi .dat siano salvati rispecchiando la struttura
            # Adatta questo path se i tuoi .dat sono altrove!
            # Esempio: root_dir/data/01/0000.dat
            dat_path = os.path.join(root_dir, line + '.dat') 
            
            # Estraiamo label dal path (es. '01' -> 1)
            parts = line.split('/')
            try:
                # Cerca la parte che è un numero di classe (es 01, 02...)
                cls_id = int(parts[-2]) # data/01/0000 -> prende 01
            except:
                cls_id = 1 # Fallback o gestione errore
            
            if cls_id in self.VALID_OBJECTS:
                self.list_of_points.append(dat_path)
                self.list_of_labels.append(cls_id)

        if verbose:
            print(f"Dataset loaded: {len(self.list_of_points)} items (Split: {split})")

        # --- NOVITÀ FUSION: Trasformazioni per l'immagine RGB (Standard ResNet) ---
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_index(self) -> List[Dict]:
        """
        Builds the list of valid samples by scanning the dataset directory.
        It performs train/test splitting and filters out missing files.
        """
        samples = []

        for obj_id in self.object_ids:
            try:
                gt_data = self.config.get_gt_data(obj_id)
                info_data = self.config.get_camera_info(obj_id)
            except FileNotFoundError:
                print(f"Warning: Data files not found for object {obj_id}")
                continue

            all_img_ids = sorted([int(k) for k in gt_data.keys()])
            if not all_img_ids:
                continue

            # Deterministic Shuffle
            rng = random.Random(self.random_seed)
            rng.shuffle(all_img_ids)

            # Train/Test Split
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
                # Robust key access (handle string vs int keys in YAML)
                annotations = gt_data.get(img_id_int) or gt_data.get(str(img_id_int)) or gt_data.get(f"{img_id_int:04d}")
                if not annotations:
                    continue

                # We only check for depth path as PointNet relies on geometry
                depth_path = obj_path / 'depth' / f"{img_id_int:04d}.png"
                if not depth_path.exists():
                    continue

                for ann in annotations:
                    actula_obj_id = int(ann['obj_id'])
                    # Ensure the annotation belongs to the current object
                    if actula_obj_id != obj_id:
                        continue

                    # Extract Pose
                    rotation_matrix = np.array(ann['cam_R_m2c']).reshape(3, 3)
                    translation_vector = np.array(ann['cam_t_m2c'])
                    quaternion_rotation = convert_rotation_to_quaternion(rotation_matrix)

                    x, y, w, h = map(int, ann['obj_bb'])
                    
                    # Basic BBox Validation
                    if w <= 0 or h <= 0: continue
                    
                    # Clip bbox to image boundaries
                    img_w, img_h = self.input_standard_dimensions
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img_w - x)
                    h = min(h, img_h - y)
                    
                    if w <= 0 or h <= 0: continue

                    bbox = [x, y, w, h]

                    # Retrieve Camera Intrinsics
                    cam_info = info_data.get(img_id_int) or info_data.get(str(img_id_int)) or info_data.get(f"{img_id_int:04d}")
                    if cam_info is None: continue

                    cam_K = np.array(cam_info['cam_K']).reshape(3, 3)

                    sample = {
                        'object_id': actula_obj_id,
                        'class_idx': self.id_to_class[actula_obj_id],
                        'img_id': img_id_int,
                        'depth_path': depth_path,
                        'rotation': quaternion_rotation,
                        'translation': translation_vector / 1000.0, # Convert mm to METERS
                        'bbox': bbox,
                        'cam_K': cam_K
                    }
                    samples.append(sample)

        return samples

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        # 1. Carica i dati geometrici (Punti, Maschera, BBox) dal .dat
        path = self.list_of_points[idx]
        with open(path, 'rb') as f:
            data = cPickle.load(f)
        
        # Estrai bounding box (rmin, rmax, cmin, cmax) salvati nel preprocessing
        # Assicurati che il tuo preprocessing salvi 'bbox' o 'rmin', etc.
        # Se non li hai salvati, devi calcolarli dalla maschera qui al volo.
        try:
            rmin, rmax, cmin, cmax = data['rmin'], data['rmax'], data['cmin'], data['cmax']
        except KeyError:
            # Fallback: ricalcola da mask se manca bbox salvata
            mask = data.get('mask', None) # Se hai la maschera
            if mask is not None:
                rows, cols = np.where(mask)
                rmin, rmax = rows.min(), rows.max()
                cmin, cmax = cols.min(), cols.max()
            else:
                # Fallback estremo se non c'è nulla (usa tutta l'immagine)
                rmin, rmax, cmin, cmax = 0, 480, 0, 640 

        # --- GESTIONE PUNTI 3D (Tuo codice originale PointNet) ---
        points = data['cld_rgb_nrm'][:, :3] # Prendi XYZ
        # ... (Tutto il tuo codice di sampling points e data augmentation va qui) ...
        # Per semplicità copio il sampling base:
        choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice, :]
        points = torch.from_numpy(points.astype(np.float32)).transpose(0, 1) # (3, N)

        # --- GESTIONE GROUND TRUTH ---
        rot = torch.from_numpy(data['rotation'].astype(np.float32))
        trans = torch.from_numpy(data['translation'].astype(np.float32))
        obj_id = self.list_of_labels[idx]
        
        # --- NOVITÀ FUSION: CARICAMENTO IMMAGINE RGB ---
        # Ricostruiamo il path dell'immagine partendo dal path del .dat
        # Se dat_path è ".../data/01/0000.dat", l'immagine è ".../data/01/rgb/0000.png"
        # Devi adattare questa stringa alla TUA struttura di cartelle esatta
        
        # Esempio generico: Sostituisci estensione e aggiungi cartella rgb se serve
        img_path = path.replace('.dat', '.png') 
        if not os.path.exists(img_path):
             # Prova jpg
             img_path = path.replace('.dat', '.jpg')
             
        # Se le immagini sono in una sottocartella 'rgb' (struttura LineMod standard)
        if not os.path.exists(img_path):
            folder, filename = os.path.split(path)
            # folder = .../data/01
            img_path = os.path.join(folder, 'rgb', filename.replace('.dat', '.png'))

        # Carica, Croppa e Trasforma
        if os.path.exists(img_path):
            rgb_img = Image.open(img_path).convert("RGB")
            # CROP usando la STESSA Bounding Box dei punti
            rgb_crop = rgb_img.crop((cmin, rmin, cmax, rmax)) # (left, top, right, bottom)
            rgb_crop = rgb_crop.resize((224, 224)) # Resize standard per ResNet
            rgb_tensor = self.img_transform(rgb_crop)
        else:
            # Immagine nera di fallback se non trova il file (per evitare crash)
            rgb_tensor = torch.zeros((3, 224, 224))

        # Ritorna Dizionario
        return {
            'points': points,          # (3, 1024)
            'images': rgb_tensor,      # (3, 224, 224) <- NUOVO CAMPO
            'rotation': rot,
            'gt_translation': trans,
            'centroid': torch.from_numpy(data['mean'].astype(np.float32)), # Se lo usi
            'object_id': obj_id
        }
    
    def get_image_path(self, idx: int):
        """Get the image path for a given sample index."""
        sample = self.samples[idx]
        obj_folder = f"{sample['object_id']:02d}"
        img_path = self.data_dir / obj_folder / 'rgb' / f"{sample['img_id']:04d}.png"
        return str(img_path)