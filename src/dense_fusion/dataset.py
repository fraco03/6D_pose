import torch
from torch.utils.data import Dataset
import os
import _pickle as cPickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class DenseFusionLineModDataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, augment=False, verbose=True):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        
        self.list_of_points = []
        self.list_of_labels = []
        
        # Mappa oggetti validi
        self.id_to_class = {
            1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat',
            8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue',
            12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'
        }
        self.VALID_OBJECTS = list(self.id_to_class.keys())

        # --- CARICAMENTO FILE (Con Fallback Intelligente) ---
        self._load_split_file(verbose)

        # --- TRASFORMAZIONI PER DENSE FUSION (RGB) ---
        # Normalizzazione standard per ResNet
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_split_file(self, verbose):
        """
        Prova a caricare la lista dei file. Se il file globale manca, 
        cerca dentro le cartelle dei singoli oggetti.
        """
        # TENTATIVO 1: File Globale (es. root/train.txt)
        global_split_file = os.path.join(self.root_dir, f'{self.split}.txt')
        file_names = []
        
        if os.path.exists(global_split_file):
            if verbose: print(f"📄 Found global split file: {global_split_file}")
            with open(global_split_file, 'r') as f:
                file_names = [x.strip() for x in f.readlines()]
        else:
            # TENTATIVO 2: Scansione cartelle oggetti (es. root/data/01/train.txt)
            if verbose: print(f"⚠️ Global file missing. Scanning object folders for '{self.split}'...")
            for oid in self.VALID_OBJECTS:
                # Cerca in: root/data/01/train.txt
                # Nota: Assumiamo che la cartella 'data' esista. 
                # Se i tuoi dati sono direttamente in root, togli 'data'.
                obj_split_path = os.path.join(self.root_dir, 'data', f'{oid:02d}', f'{self.split}.txt')
                
                if os.path.exists(obj_split_path):
                    with open(obj_split_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            clean_line = line.strip()
                            # Normalizziamo il percorso per farlo diventare: data/01/0000
                            if not clean_line.startswith("data/"):
                                clean_line = f"data/{oid:02d}/{clean_line}"
                            file_names.append(clean_line)
        
        # Processa la lista trovata
        for line in file_names:
            # line è tipo "data/01/0000"
            dat_path = os.path.join(self.root_dir, line + '.dat')
            
            # Estrazione ID (per sicurezza)
            try:
                # data/01/0000 -> prende 01
                parts = line.split('/')
                # Cerca l'ID numerico nel path
                cls_id = 0
                for p in parts:
                    if p.isdigit() and int(p) in self.VALID_OBJECTS:
                        cls_id = int(p)
                        break
            except:
                continue

            # Aggiungi solo se il file esiste e l'ID è valido
            if cls_id in self.VALID_OBJECTS and os.path.exists(dat_path):
                self.list_of_points.append(dat_path)
                self.list_of_labels.append(cls_id)
        
        if verbose:
            print(f"✅ Dataset loaded: {len(self.list_of_points)} items (Split: {self.split})")

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        path = self.list_of_points[idx]
        obj_id = self.list_of_labels[idx]
        
        # 1. CARICA IL FILE .DAT (Geometria + Metadati)
        with open(path, 'rb') as f:
            data = cPickle.load(f)
        
        # 2. ESTRAZIONE PUNTI 3D (Logica Originale PointNet)
        points = data['cld_rgb_nrm'][:, :3] # XYZ
        
        # Data Augmentation (Jittering)
        if self.augment:
            noise = np.random.normal(0, 0.005, points.shape)
            points = points + noise
            
        # Sampling (1024 punti)
        choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice, :]
        points_tensor = torch.from_numpy(points.astype(np.float32)).transpose(0, 1)

        # 3. GESTIONE BBOX (Serve per ritagliare l'immagine)
        try:
            rmin, rmax, cmin, cmax = data['rmin'], data['rmax'], data['cmin'], data['cmax']
        except KeyError:
            # Fallback: calcola da maschera se bbox non salvata
            mask = data.get('mask')
            if mask is not None:
                rows, cols = np.where(mask)
                rmin, rmax = rows.min(), rows.max()
                cmin, cmax = cols.min(), cols.max()
            else:
                rmin, rmax, cmin, cmax = 0, 480, 0, 640

        # 4. CARICAMENTO IMMAGINE RGB (Nuova logica DenseFusion)
        # Costruiamo il path dell'immagine partendo dal path del .dat
        # path es: .../data/01/0000.dat -> image: .../data/01/rgb/0000.png
        folder_dir, filename = os.path.split(path) 
        img_name = filename.replace('.dat', '.png')
        
        # Cerca immagine in sottocartella 'rgb' (standard) o root
        rgb_candidates = [
            os.path.join(folder_dir, 'rgb', img_name),
            os.path.join(folder_dir, img_name),
            os.path.join(folder_dir, 'rgb', img_name.replace('.png', '.jpg'))
        ]
        
        image_tensor = torch.zeros((3, 224, 224)) # Default nero in caso di errore
        
        for rgb_path in rgb_candidates:
            if os.path.exists(rgb_path):
                try:
                    rgb_img = Image.open(rgb_path).convert("RGB")
                    # Crop usando la Bbox (Focus sull'oggetto)
                    crop = rgb_img.crop((cmin, rmin, cmax, rmax))
                    # Resize a 224x224 per ResNet
                    crop = crop.resize((224, 224))
                    image_tensor = self.img_transform(crop)
                    break
                except Exception:
                    continue

        # 5. METADATA (Originali)
        rot = torch.from_numpy(data['rotation'].astype(np.float32))
        trans = torch.from_numpy(data['translation'].astype(np.float32))
        centroid = torch.from_numpy(data['mean'].astype(np.float32))

        # 6. OUTPUT (Compatibile sia con PointNet che DenseFusion)
        return {
            'points': points_tensor,       # (3, 1024) -> Per PointNet
            'images': image_tensor,        # (3, 224, 224) -> Per ResNet (NUOVO)
            'rotation': rot,               # GT Rot
            'gt_translation': trans,       # GT Trans
            'centroid': centroid,          # Centroide
            'object_id': obj_id,           # ID Classe
            'label': obj_id                # Alias per retro-compatibilità
        }