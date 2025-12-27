import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm



# Mapping: Linemod Folder Name -> YOLO Class ID
# Assumendo che le cartelle siano '1', '2'... e YOLO sia 0, 1...
# Modifica questo dizionario se i tuoi ID sono diversi
LINEMOD_TO_YOLO_ID = {
    '1': 0,   # ape
    '2': 1,   # benchvise
    # '3': bowl (non presente nel tuo yolo, quindi lo saltiamo)
    '4': 2,   # camera
    '5': 3,   # can
    '6': 4,   # cat
    # '7': cup (non presente nel tuo yolo)
    '8': 5,   # driller
    '9': 6,   # duck
    '10': 7,  # eggbox (Conferma: Folder 10 -> Yolo 7)
    '11': 8,  # glue
    '12': 9,  # holepuncher
    '13': 10, # iron
    '14': 11, # lamp
    '15': 12  # phone
}

def load_yaml(path):
    with open(path, 'r') as f:
        # Loader sicuro per evitare esecuzione di codice arbitrario
        return yaml.load(f, Loader=yaml.SafeLoader)

def save_yaml(data, path):
    with open(path, 'w') as f:
        # default_flow_style=None mantiene un formato leggibile
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)

def main(LINEMOD_ROOT: Path, OUTPUT_ROOT: Path, MODEL_PATH: str, CONF_THRESHOLD: float):
    # 1. Carica il modello
    print(f"Caricamento modello YOLO da {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 2. Itera sulle cartelle degli oggetti (es. '1', '2', etc.)
    obj_folders = [f for f in os.listdir(LINEMOD_ROOT) if (LINEMOD_ROOT / f).is_dir()]
    
    # Ordiniamo convertendo a int per sicurezza (così 10 viene dopo 2, e non dopo 1)
    obj_folders.sort(key=lambda x: int(x) if x.isdigit() else x)

    for obj_folder in obj_folders:
        # TRUCCO: Convertiamo il nome cartella in int e poi di nuovo in stringa.
        # In questo modo:
        # "01" diventa 1 -> "1" (trova la chiave nel dizionario)
        # "1"  diventa 1 -> "1" (funziona uguale)
        # "10" diventa 10 -> "10"
        try:
            key_lookup = str(int(obj_folder))
        except ValueError:
            # Se c'è una cartella che non è un numero (es. "info"), la saltiamo
            continue

        # Verifica se abbiamo un mapping per questo oggetto usando la chiave normalizzata
        if key_lookup not in LINEMOD_TO_YOLO_ID:
            print(f"Skipping cartella '{obj_folder}' (Key: {key_lookup}): ID non presente nel mapping.")
            continue

        target_yolo_class = LINEMOD_TO_YOLO_ID[key_lookup]
        
        # Percorsi (Nota: per il path usiamo obj_folder originale che può essere "01")
        input_dir = LINEMOD_ROOT / obj_folder
        output_dir = OUTPUT_ROOT / obj_folder # Creerà output/01/yolo.yml
        gt_path = input_dir / 'gt.yml'
        
        if not gt_path.exists():
            print(f"File gt.yml non trovato in {obj_folder}")
            continue

        # Crea directory di destinazione
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processando oggetto: {obj_folder} (Key: {key_lookup} -> YOLO Class: {target_yolo_class})")
        
        # Carica GT originale
        gt_data = load_yaml(gt_path)
        yolo_data = {} # Nuovo dizionario per yolo.yaml

        # Itera su ogni frame presente nel gt.yml
        # gt_data è strutturato come { frame_id: [ {obj_data}, ... ] }
        for frame_id, annotations in tqdm(gt_data.items(), desc=f"Frames Obj {obj_folder}"):
            
            # Costruisci path immagine (assumendo formato 0000.png)
            img_name = f"{frame_id:04d}.png"
            img_path = input_dir / 'rgb' / img_name
            
            if not img_path.exists():
                # Fallback se l'immagine non esiste (raro)
                yolo_data[frame_id] = annotations
                continue

            # Esegui Inferenza YOLO
            results = model.predict(source=str(img_path), conf=CONF_THRESHOLD, verbose=False)
            
            # Prepara la lista di annotazioni per questo frame (copia profonda per sicurezza)
            new_annotations = []
            
            # Nota: Linemod spesso ha 1 oggetto per frame, ma la struttura è una lista.
            # Qui assumiamo di processare l'annotazione corrispondente all'oggetto della cartella.
            
            for ann in annotations:
                # Copia i dati originali (cam_R, cam_t, obj_id)
                new_ann = ann.copy()
                original_bb = ann['obj_bb'] # [x, y, w, h]
                
                found_yolo = False
                best_conf = -1
                best_box = None

                # Cerca tra le detection di YOLO quella corretta
                if len(results) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Se la classe combacia con l'oggetto corrente
                        if cls_id == target_yolo_class:
                            # Se ci sono più istanze, prendiamo quella con confidenza maggiore
                            if conf > best_conf:
                                best_conf = conf
                                # YOLO ritorna xyxy, salviamo per dopo
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                # Converti in xywh per formato Linemod
                                w = x2 - x1
                                h = y2 - y1
                                best_box = [int(x1), int(y1), int(w), int(h)]
                                found_yolo = True

                # Logica di assegnazione
                if found_yolo and best_box is not None:
                    new_ann['obj_bb'] = best_box
                    new_ann['is_yolo'] = True
                    # Opzionale: salva anche la confidence
                    new_ann['yolo_conf'] = float(round(best_conf, 4))
                else:
                    # FALLBACK A GT
                    new_ann['obj_bb'] = original_bb
                    new_ann['is_yolo'] = False
                
                new_annotations.append(new_ann)
            
            # Salva nel dizionario finale
            yolo_data[frame_id] = new_annotations

        # Scrivi yolo.yaml nella nuova destinazione
        save_path = output_dir / 'yolo.yml'
        save_yaml(yolo_data, save_path)
        print(f"Salvato: {save_path}")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO-based labels for Linemod dataset.")
    parser.add_argument('--linemod_root', type=str, required=True, help="Path to the Linemod dataset root.", default='/kaggle/input/line-mode/Linemod_preprocessed/data')
    parser.add_argument('--output_root', type=str, required=True, help="Path to save the generated YOLO labels. (yolo.yml files)", default='/kaggle/working/Linemod_YOLO')
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained YOLO model.", default='/kaggle/input/yolo-student/pytorch/default/1/yolo11s_autolabel_final_with_80_th/weights/best.pt')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help="Confidence threshold for YOLO detections.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    LINEMOD_ROOT = Path(args.linemod_root)
    OUTPUT_ROOT = Path(args.output_root)
    MODEL_PATH = args.model_path
    CONF_THRESHOLD = args.conf_threshold

    main(LINEMOD_ROOT, OUTPUT_ROOT, MODEL_PATH, CONF_THRESHOLD)