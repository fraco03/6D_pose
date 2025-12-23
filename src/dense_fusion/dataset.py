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
        
        self.list_of_points = []
        self.list_of_labels = []
        
        self.id_to_class = {
            1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat',
            8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue',
            12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'
        }
        self.VALID_OBJECTS = list(self.id_to_class.keys())

        # --- LOGICA ORIGINALE DI CARICAMENTO ---
        # Si fida del fatto che esista un file .txt valido nella root
        split_file = os.path.join(root_dir, f'{split}.txt')
        
        if not os.path.exists(split_file):
            print(f"❌ Critical Error: Split file not found at {split_file}")
            # Se manca, provo un fallback comune su Kaggle (nella working directory)
            fallback_path = os.path.join("/kaggle/working", f'{split}.txt')
            if os.path.exists(fallback_path):
                print(f"✅ Found fallback file at {fallback_path}")
                split_file = fallback_path

        with open(split_file, 'r') as f:
            file_names = [x.strip() for x in f.readlines()]
        
        for line in file_names:
            # line è tipo "data/01/0000"
            # Costruiamo il path completo
            dat_path = os.path.join(root_dir, line + '.dat')
            
            # Estraiamo la classe dal path (es. data/01/0000 -> 01)
            try:
                # Splitta su '/' e cerca quale parte è un ID valido
                parts = line.split('/')
                cls_id = 0
                for part in parts:
                    if part.isdigit() and int(part) in self.VALID_OBJECTS:
                        cls_id = int(part)
                        break
            except:
                continue

            # Se l'oggetto è valido, lo aggiungiamo
            if cls_id in self.VALID_OBJECTS:
                # Verifica extra: se il file non esiste in root, forse il txt punta male?
                if not os.path.exists(dat_path):
                    # Prova a cercarlo relativo alla root corrente
                    # A volte i txt hanno path assoluti vecchi
                    pass 
                
                self.list_of_points.append(dat_path)
                self.list_of_labels.append(cls_id)

        if verbose:
            print(f"Dataset loaded: {len(self.list_of_points)} items (Split: {split})")

        # --- NUOVO: Trasformazioni Immagine per ResNet ---
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        # 1. Carica il file .dat (come prima)
        path = self.list_of_points[idx]
        obj_id = self.list_of_labels[idx]
        
        try:
            with open(path, 'rb') as f:
                data = cPickle.load(f)
        except Exception:
            # Se il file è corrotto, prendine un altro a caso
            return self.__getitem__(np.random.randint(0, len(self)))
        
        # 2. Estrai Bounding Box (Serve per ritagliare l'immagine)
        try:
            rmin, rmax, cmin, cmax = data['rmin'], data['rmax'], data['cmin'], data['cmax']
        except KeyError:
            # Fallback se mancano i dati bbox
            mask = data.get('mask')
            if mask is not None:
                rows, cols = np.where(mask)
                rmin, rmax = rows.min(), rows.max()
                cmin, cmax = cols.min(), cols.max()
            else:
                rmin, rmax, cmin, cmax = 0, 480, 0, 640

        # 3. Punti 3D (Tuo codice originale)
        points = data['cld_rgb_nrm'][:, :3]
        if self.augment:
            noise = np.random.normal(0, 0.005, points.shape)
            points = points + noise
            
        choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice, :]
        points_tensor = torch.from_numpy(points.astype(np.float32)).transpose(0, 1)

        # 4. --- NUOVO: Carica e Ritaglia Immagine RGB ---
        folder_dir, filename = os.path.split(path) 
        img_basename = filename.replace('.dat', '') # "0000"
        
        # Cerchiamo l'immagine .png o .jpg (potrebbe essere in cartella 'rgb' o root)
        candidates = [
            os.path.join(folder_dir, 'rgb', img_basename + '.png'),
            os.path.join(folder_dir, img_basename + '.png'),
            os.path.join(folder_dir, 'rgb', img_basename + '.jpg'),
            os.path.join(folder_dir, img_basename + '.jpg')
        ]
        
        image_tensor = torch.zeros((3, 224, 224)) # Default nero
        
        for img_path in candidates:
            if os.path.exists(img_path):
                try:
                    rgb_img = Image.open(img_path).convert("RGB")
                    # Crop (Ritaglia solo l'oggetto usando la BBox)
                    crop = rgb_img.crop((cmin, rmin, cmax, rmax))
                    # Resize a 224x224 (Input fisso per ResNet)
                    crop = crop.resize((224, 224))
                    image_tensor = self.img_transform(crop)
                    break
                except Exception:
                    pass

        # 5. Metadata (Originali)
        rot = torch.from_numpy(data['rotation'].astype(np.float32))
        trans = torch.from_numpy(data['translation'].astype(np.float32))
        centroid = torch.from_numpy(data['mean'].astype(np.float32))

        return {
            'points': points_tensor,
            'images': image_tensor,     # <--- NUOVO CAMPO AGGIUNTO
            'rotation': rot,
            'gt_translation': trans,
            'centroid': centroid,
            'object_id': obj_id,
            'label': obj_id
        }