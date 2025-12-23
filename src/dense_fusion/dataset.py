import torch
from torch.utils.data import Dataset
import os
import _pickle as cPickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class FusionLineModDataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, augment=False, verbose=True):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        
        # Mappa oggetti
        self.id_to_class = {
            1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat',
            8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue',
            12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'
        }
        self.VALID_OBJECTS = list(self.id_to_class.keys())

        # Trasformazione Immagini (Standard ResNet)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Costruisce l'indice scansionando le cartelle (Risolve problema file mancanti)
        self.list_of_points, self.list_of_labels = self._build_index(verbose)

    def _build_index(self, verbose):
        list_of_points = []
        list_of_labels = []
        
        if verbose: print(f"🔍 Building index for split '{self.split}'...")

        for oid in self.VALID_OBJECTS:
            # Cerca: root/data/01/train.txt
            obj_folder = os.path.join(self.root_dir, "data", f"{oid:02d}")
            split_file = os.path.join(obj_folder, f"{self.split}.txt")
            
            if not os.path.exists(split_file):
                continue
                
            with open(split_file, 'r') as f:
                file_names = [x.strip() for x in f.readlines()]
            
            for name in file_names:
                # Se la linea è solo "0000", costruisci path completo
                dat_path = os.path.join(obj_folder, name + '.dat')
                if os.path.exists(dat_path):
                    list_of_points.append(dat_path)
                    list_of_labels.append(oid)
        
        if verbose:
            print(f"✅ Loaded {len(list_of_points)} items for {self.split}")
        return list_of_points, list_of_labels

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        path = self.list_of_points[idx]
        obj_id = self.list_of_labels[idx]
        
        # 1. CARICA DATI DAT (Pickle)
        with open(path, 'rb') as f:
            data = cPickle.load(f)
        
        # 2. RECUPERA BBOX (Necessario per Crop Immagine)
        try:
            rmin, rmax, cmin, cmax = data['rmin'], data['rmax'], data['cmin'], data['cmax']
        except KeyError:
            # Fallback su mask se bbox non salvata esplicitamente
            mask = data.get('mask')
            if mask is not None:
                rows, cols = np.where(mask)
                rmin, rmax = rows.min(), rows.max()
                cmin, cmax = cols.min(), cols.max()
            else:
                rmin, rmax, cmin, cmax = 0, 480, 0, 640

        # 3. PUNTI 3D (Logica Originale PointNet)
        points = data['cld_rgb_nrm'][:, :3] # XYZ
        
        if self.augment:
            # Jittering sui punti
            noise = np.random.normal(0, 0.005, points.shape)
            points = points + noise
            
        # Sampling punti
        choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice, :]
        points_tensor = torch.from_numpy(points.astype(np.float32)).transpose(0, 1) # (3, 1024)

        # 4. IMMAGINE RGB (Nuova logica per Fusion)
        folder_dir, filename = os.path.split(path) 
        img_name = filename.replace('.dat', '.png')
        
        # Cerca immagine in 'rgb/' o root
        rgb_candidates = [
            os.path.join(folder_dir, 'rgb', img_name),
            os.path.join(folder_dir, img_name),
            os.path.join(folder_dir, 'rgb', img_name.replace('.png', '.jpg'))
        ]
        
        image_tensor = torch.zeros((3, 224, 224)) # Default nero
        
        for rgb_path in rgb_candidates:
            if os.path.exists(rgb_path):
                try:
                    rgb_img = Image.open(rgb_path).convert("RGB")
                    # Crop (usando bbox) e Resize
                    crop = rgb_img.crop((cmin, rmin, cmax, rmax))
                    crop = crop.resize((224, 224))
                    image_tensor = self.img_transform(crop)
                    break
                except Exception:
                    pass

        # 5. METADATA (Originali)
        rot = torch.from_numpy(data['rotation'].astype(np.float32))
        trans = torch.from_numpy(data['translation'].astype(np.float32))
        centroid = torch.from_numpy(data['mean'].astype(np.float32))

        # 6. RITORNA TUTTO (Struttura compatibile con Evaluation)
        return {
            'points': points_tensor,       # Usato da PointNet
            'images': image_tensor,        # NUOVO: Usato da DenseFusion
            'rotation': rot,               # Usato da Loss/Eval
            'gt_translation': trans,       # Usato da Loss/Eval
            'centroid': centroid,          # Usato da Eval
            'object_id': obj_id            # Usato da Eval
        }