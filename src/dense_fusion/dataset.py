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
        
        # Mapping oggetti (Standard LineMod)
        self.id_to_class = {
            1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat',
            8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue',
            12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'
        }
        self.VALID_OBJECTS = list(self.id_to_class.keys())

        # Trasformazioni per l'immagine RGB (Standard per ResNet)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- COSTRUZIONE INDICE ROBUSTA ---
        # Usa _build_index per trovare i file ovunque essi siano
        self.list_of_points, self.list_of_labels = self._build_index(verbose)

    def _build_index(self, verbose):
        """
        Scansiona le directory per trovare i file di split e i file .dat corrispondenti.
        Risolve automaticamente i problemi di percorso (data/ vs root/).
        """
        list_of_points = []
        list_of_labels = []
        
        if verbose: print(f"🔍 Building index for split '{self.split}' in: {self.root_dir}")

        # Verifica preliminare struttura cartelle
        has_data_folder = os.path.exists(os.path.join(self.root_dir, 'data'))
        base_search_path = os.path.join(self.root_dir, 'data') if has_data_folder else self.root_dir

        for oid in self.VALID_OBJECTS:
            # Cerca la cartella dell'oggetto: es. .../data/01 o .../1
            # Proviamo con padding (01) e senza (1)
            obj_folder_candidates = [
                os.path.join(base_search_path, f"{oid:02d}"), # 01
                os.path.join(base_search_path, f"{oid}")      # 1
            ]
            
            obj_folder = None
            for p in obj_folder_candidates:
                if os.path.exists(p):
                    obj_folder = p
                    break
            
            if obj_folder is None:
                continue # Oggetto non trovato in questa directory

            # Cerca il file split (train.txt / test.txt) dentro la cartella oggetto
            split_file = os.path.join(obj_folder, f"{self.split}.txt")
            
            if not os.path.exists(split_file):
                continue # File txt mancante per questo oggetto
                
            # Leggi i file
            with open(split_file, 'r') as f:
                file_names = [x.strip() for x in f.readlines()]
            
            added_count = 0
            for name in file_names:
                # name potrebbe essere "0000" oppure "data/01/0000"
                # Puliamo il nome per avere solo il filename base
                clean_name = name.split('/')[-1] # Prende "0000"
                
                # Costruiamo il path assoluto al file .dat
                dat_path = os.path.join(obj_folder, clean_name + '.dat')
                
                if os.path.exists(dat_path):
                    list_of_points.append(dat_path)
                    list_of_labels.append(oid)
                    added_count += 1
            
            # Debug per capire se sta trovando roba
            # if verbose and added_count > 0:
            #    print(f"   Object {oid:02d}: found {added_count} items")

        if verbose:
            print(f"✅ Total items loaded: {len(list_of_points)}")
            if len(list_of_points) == 0:
                print("❌ ERROR: 0 items loaded. Check if dataset_root points to the folder containing 'data' or the objects.")
            
        return list_of_points, list_of_labels

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        path = self.list_of_points[idx]
        obj_id = self.list_of_labels[idx]
        
        # 1. CARICAMENTO DATI GEOMETRICI (.dat)
        # Usiamo un try-except per evitare crash su file corrotti
        try:
            with open(path, 'rb') as f:
                data = cPickle.load(f)
        except Exception as e:
            # Se fallisce, prendiamo un altro indice a caso
            return self.__getitem__(np.random.randint(0, len(self)))

        # 2. BOUNDING BOX (Serve per ritagliare l'immagine)
        try:
            rmin, rmax, cmin, cmax = data['rmin'], data['rmax'], data['cmin'], data['cmax']
        except KeyError:
            # Fallback usando la maschera
            mask = data.get('mask')
            if mask is not None:
                rows, cols = np.where(mask)
                rmin, rmax = rows.min(), rows.max()
                cmin, cmax = cols.min(), cols.max()
            else:
                rmin, rmax, cmin, cmax = 0, 480, 0, 640

        # 3. PUNTI 3D (Logica PointNet)
        points = data['cld_rgb_nrm'][:, :3] # XYZ
        
        if self.augment:
            noise = np.random.normal(0, 0.005, points.shape)
            points = points + noise
            
        choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice, :]
        points_tensor = torch.from_numpy(points.astype(np.float32)).transpose(0, 1)

        # 4. RGB IMAGE LOADING & CROP (Logica DenseFusion)
        # Il path è assoluto al file .dat: .../data/01/0000.dat
        folder_dir, filename = os.path.split(path) 
        img_basename = filename.replace('.dat', '') # "0000"
        
        # Cerca l'immagine in vari posti possibili
        rgb_path_candidates = [
            os.path.join(folder_dir, 'rgb', img_basename + '.png'), # Standard LineMod: rgb/0000.png
            os.path.join(folder_dir, img_basename + '.png'),        # Direttamente nella cartella
            os.path.join(folder_dir, 'rgb', img_basename + '.jpg'), # Formato JPG
            os.path.join(folder_dir, img_basename + '.jpg')
        ]
        
        image_tensor = torch.zeros((3, 224, 224)) # Default nero (fallback)
        
        for p in rgb_path_candidates:
            if os.path.exists(p):
                try:
                    rgb_img = Image.open(p).convert("RGB")
                    # Crop (usando bbox) e Resize
                    crop = rgb_img.crop((cmin, rmin, cmax, rmax))
                    crop = crop.resize((224, 224))
                    image_tensor = self.img_transform(crop)
                    break # Trovata!
                except Exception:
                    pass

        # 5. METADATA
        rot = torch.from_numpy(data['rotation'].astype(np.float32))
        trans = torch.from_numpy(data['translation'].astype(np.float32))
        centroid = torch.from_numpy(data['mean'].astype(np.float32))

        # 6. OUTPUT COMPATIBILE
        return {
            'points': points_tensor,   # (3, 1024) PointNet
            'images': image_tensor,    # (3, 224, 224) DenseFusion
            'rotation': rot,
            'gt_translation': trans,
            'centroid': centroid,
            'object_id': obj_id,
            'label': obj_id
        }