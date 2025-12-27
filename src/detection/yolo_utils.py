import os
import yaml
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import cv2
import os
import matplotlib.pyplot as plt
import random
import shutil
import requests
from collections import defaultdict


def compute_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x1, y1, x2, y2] format"""
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

import json

def load_test_dataset_from_disk(load_path='/kaggle/working/test_data_subset.json'):
    print(f"📂 Caricamento dati da: {load_path}")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError("❌ Il file JSON non esiste! Caricalo nello spazio di lavoro.")
        
    with open(load_path, 'r') as f:
        data = json.load(f)
        
    print(f"✅ Caricati {len(data)} elementi.")
    return data

# CHANGE iou FOR DIFFERENET mAP METRICS
def calculate_adapted_map50(model_path, test_data_input, iou_thresh=0.5):
    """
    load data from load_test_dataset_from_disk() and pass it here to
    Calculate adapted mAP (masked evaluation) using the list of test data dictionary.
    
    Strategy:
    1. Extracts the 'Target Class' from the source file path (e.g., .../01/rgb/.. -> Class 1).
    2. Filters Ground Truth: Only considers labels belonging to that Target Class.
    3. Filters Predictions: Only considers predictions of that Target Class.
    4. Ignores all other objects (avoiding False Positives for unlabelled background objects).

    Args:
        model_path (str): Path to trained YOLO model weights.
        test_data_input (list): The GLOBAL_TEST_SUBSET list from Step 1.
        iou_thresh (float): IoU threshold (default 0.5).

    Returns:
        float: Mean Average Precision (mAP).
    """
    
    # --- CONFIGURATION ---
    # Must match the training class mapping
    valid_obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    class_names = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 
                   'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
    
    # Map Real Folder ID (1, 2...) -> YOLO Class Index (0, 1...)
    id_map = {obj_id: i for i, obj_id in enumerate(valid_obj_ids)}

    print(f"⚖️  Computing adapted mAP@{int(iou_thresh*100)} (Masked Evaluation)...")
    model = YOLO(model_path)
    
    # Data structures for AP calculation per class
    # class_preds[class_idx] = list of [confidence, is_true_positive (1 or 0)]
    class_preds = {i: [] for i in range(len(class_names))} 
    class_total_gt = {i: 0 for i in range(len(class_names))}

    # --- HELPER: IoU Calculation ---
    def compute_iou(boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        return interArea / unionArea if unionArea > 0 else 0

    # --- MAIN EVALUATION LOOP ---
    for item in tqdm(test_data_input, desc="Evaluating"):
        
        src_path = item['src']
        labels_raw = item['labels']
        
        # 1. DETERMINE TARGET CLASS from File Path
        # We assume path structure like: .../{folder_id}/rgb/xxxx.png
        # We split path and find the folder name
        try:
            # Get parent of parent folder name (assuming .../01/rgb/img.png)
            folder_str = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
            folder_id = int(folder_str)
            
            if folder_id not in id_map: continue # Skip if folder is not in our list
            
            target_cls_idx = id_map[folder_id]
        except:
            continue # Skip if path format is unexpected

        # 2. PARSE GROUND TRUTH (Filter by Target Class)
        img = cv2.imread(src_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        gt_boxes = []
        for lbl in labels_raw:
            parts = list(map(float, lbl.split()))
            cls_id = int(parts[0])
            
            # ADAPTED STRATEGY: Only keep GT if it matches the folder class
            if cls_id == target_cls_idx:
                # YOLO format (xc, yc, w, h) normalized -> Absolute (x1, y1, x2, y2)
                xc, yc, w, h = parts[1], parts[2], parts[3], parts[4]
                x1 = (xc - w/2) * w_img
                y1 = (yc - h/2) * h_img
                x2 = (xc + w/2) * w_img
                y2 = (yc + h/2) * h_img
                gt_boxes.append([x1, y1, x2, y2])

        if not gt_boxes: continue # No relevant object in this image

        class_total_gt[target_cls_idx] += len(gt_boxes)

        # 3. RUN PREDICTION
        # conf=0.01 is crucial to get the full Precision-Recall curve
        results = model.predict(img, conf=0.01, verbose=False)
        
        # 4. FILTER PREDICTIONS (Filter by Target Class)
        valid_preds = []
        for box in results[0].boxes:
            p_cls = int(box.cls[0])
            
            # ADAPTED STRATEGY: Ignore predictions of other classes
            if p_cls == target_cls_idx:
                # Get coordinates
                x, y, w, h = box.xywh[0].tolist() # xywh is absolute center
                x1, y1 = x - w/2, y - h/2
                x2, y2 = x + w/2, y + h/2
                conf = float(box.conf[0])
                valid_preds.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

        # 5. MATCHING (IoU)
        # Sort predictions by confidence (High -> Low)
        valid_preds.sort(key=lambda x: x['conf'], reverse=True)
        gt_matched = [False] * len(gt_boxes)
        
        for pred in valid_preds:
            best_iou = 0
            best_gt_idx = -1
            
            # Find best overlapping GT
            for idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred['bbox'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # Check Threshold
            if best_iou >= iou_thresh:
                if not gt_matched[best_gt_idx]:
                    # True Positive
                    class_preds[target_cls_idx].append([pred['conf'], 1])
                    gt_matched[best_gt_idx] = True
                else:
                    # False Positive (Duplicate detection)
                    class_preds[target_cls_idx].append([pred['conf'], 0])
            else:
                # False Positive (Low IoU or Background)
                class_preds[target_cls_idx].append([pred['conf'], 0])

    # --- CALCULATE AP PER CLASS ---
    print("\n" + "="*60)
    print(f"{'CLASS':<15} | {'AP@' + str(int(iou_thresh*100)):<10} | {'GT Count':<10} | {'Preds Count':<10}")
    print("-" * 60)
    
    aps = []
    
    for cls_idx in range(len(class_names)):
        preds = class_preds[cls_idx]
        total_gt = class_total_gt[cls_idx]
        
        if total_gt == 0:
            print(f"{class_names[cls_idx]:<15} | {'N/A':<10} | {0:<10} | {len(preds)}")
            continue
            
        if not preds:
            ap = 0.0
        else:
            preds = np.array(preds)
            # sort by confidence descending (already roughly sorted but good safety)
            sort_ind = np.argsort(-preds[:, 0])
            preds = preds[sort_ind]
            
            tp = np.cumsum(preds[:, 1])
            fp = np.cumsum(1 - preds[:, 1])
            
            rec = tp / total_gt
            prec = tp / (tp + fp + 1e-16)
            
            # 11-point interpolation or AUC (Area Under Curve)
            # We use AUC integration with padding
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            
            # Monotonically decreasing precision
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
                
            # Area under curve
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            
        aps.append(ap)
        print(f"{class_names[cls_idx]:<15} | {ap:.4f}     | {total_gt:<10} | {len(preds)}")
    
    mean_ap = np.mean(aps) if aps else 0.0
    print("-" * 60)
    print(f"Adapted mAP@{int(iou_thresh*100)}  | {mean_ap:.4f}")
    print("="*60)
    
    return mean_ap
   
def visualize_bbox(image_path):
    """
    Visualize YOLO Bounding Boxes on a specific image
    
    Args:
        image_path: Path to the image file
    """

    # 1. Define Classes (same order as data.yaml)
    class_names = [
        'ape', 'benchvise', 'camera', 'can', 'cat',
        'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
        'iron', 'lamp', 'phone'
    ]

    # Generate random colors for each class to distinguish them
    np.random.seed(42)
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))

    # 2. Load Image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Unable to read image file.")
        return

    h_img, w_img, _ = img.shape

    # 3. Derive Label Path
    # Replace 'images' with 'labels' and extension with .txt
    label_path = image_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'

    if not os.path.exists(label_path):
        print(f"⚠️ Warning: No label file found for this image ({label_path})")
        # Show the empty image anyway
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        return

    # 4. Read and Draw Boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()

    print(f"✅ Found {len(lines)} objects in the image.")

    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls_id = int(parts[0])
        x_c, y_c, w, h = parts[1], parts[2], parts[3], parts[4]

        # Convert from YOLO (normalized) to Pixel coordinates
        x1 = int((x_c - w / 2) * w_img)
        y1 = int((y_c - h / 2) * h_img)
        x2 = int((x_c + w / 2) * w_img)
        y2 = int((y_c + h / 2) * h_img)

        # Get name and color
        label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        color = colors[cls_id]

        # Draw Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw Label with colored background
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 5. Visualization (Matplotlib is better on Colab than cv2.imshow)
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Visualization: {os.path.basename(image_path)}")
    plt.show()

def create_teacher_dataset_final(source_root, dest_root, bg_cache_dir, num_collages=2000, max_objects_images=4):
    # Object IDs and class mapping (excluding objects 3 and 7)
    valid_obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    id_map = {obj_id: i for i, obj_id in enumerate(valid_obj_ids)}
    obj_folders = [f"{i:02d}" for i in valid_obj_ids]
    class_names = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']

    # --- SETUP DIRECTORIES ---
    print("🛠️  Setting up directories...")
    if os.path.exists(dest_root): shutil.rmtree(dest_root)
    if not os.path.exists(bg_cache_dir): os.makedirs(bg_cache_dir)

    # Create only 'train' and 'test' directories (80/20 split)
    for split in ['train', 'test']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(dest_root, dtype, split), exist_ok=True)

    # --- DOWNLOAD BACKGROUNDS ---
    existing_bgs = [f for f in os.listdir(bg_cache_dir) if f.endswith('.jpg')]
    if len(existing_bgs) < 50:
        print("🌍 Downloading background images...")
        for i in range(50 - len(existing_bgs)):
            try:
                r = requests.get("https://picsum.photos/640/640", timeout=5)
                if r.status_code == 200:
                    with open(os.path.join(bg_cache_dir, f"bg_{i}.jpg"), "wb") as f:
                        f.write(r.content)
            except: pass
    bg_files = [os.path.join(bg_cache_dir, f) for f in os.listdir(bg_cache_dir) if f.endswith('.jpg')]
    if not bg_files: raise RuntimeError("❌ No background images found!")

    # --- HELPER FUNCTIONS ---
    def rotate_image(image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # --- DATA COLLECTION & SPLITTING (80% Train / 20% Test) ---
    print("📊 Analyzing and splitting data (80% Train / 20% Test)...")
    
    # Store training samples for collage generation (for non-02 classes)
    train_objects_cache = []
    
    # Statistics
    stats = {'train_02_real': 0, 'train_other_cached': 0, 'test_real': 0}
    train_subset_all = []
    test_subset_all = []

    for folder_str in tqdm(obj_folders, desc="Processing Classes"):
        class_id = id_map[int(folder_str)]
        is_class_02 = (int(folder_str) == 2)
        base_dir = os.path.join(source_root, folder_str)
        gt_path = os.path.join(base_dir, 'gt.yml')
        
        if not os.path.exists(gt_path): continue
        
        # 1. Load GT Data
        with open(gt_path, 'r') as f: gt_data = yaml.safe_load(f)
        
        # 2. Collect all valid images for this class
        all_images = []
        for img_id_int, anns in gt_data.items():
            fname = f"{img_id_int:04d}.png"
            src_img = os.path.join(base_dir, 'rgb', fname)
            mask_path = os.path.join(base_dir, 'mask', fname)
            
            if not os.path.exists(src_img): continue

            # Prepare YOLO labels with CLAMPING
            yolo_labels = []
            img_w, img_h = 640, 480  # Standard LineMod dimensions

            for ann in anns:
                oid = int(ann['obj_id'])
                if oid in id_map:
                    x, y, w, h = ann['obj_bb']
                    
                    # 
                    # --- CRITICAL FIX: Clamp coordinates to image boundaries ---
                    # Sometimes GT boxes are slightly outside (e.g., -5 or 645)
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(img_w, x + w)
                    y2 = min(img_h, y + h)
                    
                    # Recalculate width/height after clamping
                    w_clamped = x2 - x1
                    h_clamped = y2 - y1
                    
                    # Only add label if the box is still valid (> 1 pixel)
                    if w_clamped > 1 and h_clamped > 1:
                        # Normalize to 0.0 - 1.0
                        xc = (x1 + w_clamped/2) / img_w
                        yc = (y1 + h_clamped/2) / img_h
                        wn = w_clamped / img_w
                        hn = h_clamped / img_h
                        
                        # Extra safety clip (floating point errors)
                        xc = np.clip(xc, 0.0, 1.0)
                        yc = np.clip(yc, 0.0, 1.0)
                        wn = np.clip(wn, 0.0, 1.0)
                        hn = np.clip(hn, 0.0, 1.0)

                        yolo_labels.append(f"{id_map[oid]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            
            if not yolo_labels: continue

            # Store image data
            all_images.append({
                'src': src_img,
                'mask': mask_path,
                'fname': fname,
                'labels': yolo_labels
            })

        # 3. Shuffle and Split 80/20
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.80)
        
        train_subset = all_images[:split_idx]
        test_subset = all_images[split_idx:]
        
        train_subset_all.extend(train_subset)   
        test_subset_all.extend(test_subset)
        
        # --- HANDLE TRAINING SUBSET (80%) ---
        if is_class_02:
            # Class 02: Save real images directly to train folder
            for img_data in train_subset:
                dst_fname = f"real_{folder_str}_{img_data['fname']}"
                shutil.copy(img_data['src'], os.path.join(dest_root, 'images/train', dst_fname))
                with open(os.path.join(dest_root, 'labels/train', dst_fname.replace('.png','.txt')), 'w') as f:
                    f.write('\n'.join(img_data['labels']))
                stats['train_02_real'] += 1
        else:
            # Other Classes: Cache images/masks for collage generation
            # Limit to 50 samples to prevent memory overflow, but randomly selected from the 80% split
            collage_samples = train_subset
            
            for img_data in collage_samples:
                if os.path.exists(img_data['mask']):
                    img = cv2.imread(img_data['src'])
                    mask = cv2.imread(img_data['mask'], cv2.IMREAD_GRAYSCALE)
                    
                    if img is None or mask is None: continue

                    x, y, w, h = cv2.boundingRect(mask)
                    if w > 5 and h > 5:
                        crop_img = img[y:y+h, x:x+w]
                        crop_mask = mask[y:y+h, x:x+w]
                        _, crop_mask = cv2.threshold(crop_mask, 127, 255, cv2.THRESH_BINARY)
                        train_objects_cache.append({'cls': class_id, 'img': crop_img, 'mask': crop_mask})
                        stats['train_other_cached'] += 1

        # --- HANDLE TEST SUBSET (20%) ---
        for img_data in test_subset:
            dst_fname = f"real_{folder_str}_{img_data['fname']}"
            shutil.copy(img_data['src'], os.path.join(dest_root, 'images/test', dst_fname))
            with open(os.path.join(dest_root, 'labels/test', dst_fname.replace('.png','.txt')), 'w') as f:
                f.write('\n'.join(img_data['labels']))
            stats['test_real'] += 1

    # --- GENERATE SYNTHETIC COLLAGES (Using only Training Data) ---
    print(f"🚀 Generating {num_collages} synthetic collages...")
    if not train_objects_cache: raise RuntimeError("❌ Training cache empty!")

    for i in tqdm(range(num_collages), desc="Synthesizing"):
        bg = cv2.imread(random.choice(bg_files))
        bg = cv2.resize(bg, (640, 640))
        h_bg, w_bg = bg.shape[:2]
        
        labels = []
        occupied_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
        
        for _ in range(random.randint(1, max_objects_images)):
            obj = random.choice(train_objects_cache)
            img_aug, mask_aug = obj['img'].copy(), obj['mask'].copy()
            
            angle = random.randint(-180, 180)
            img_aug = rotate_image(img_aug, angle)
            mask_aug = rotate_image(mask_aug, angle)
            
            scale = random.uniform(0.4, 0.9)
            nh, nw = int(img_aug.shape[0] * scale), int(img_aug.shape[1] * scale)
            if nh <= 0 or nw <= 0: continue
            img_aug = cv2.resize(img_aug, (nw, nh))
            mask_aug = cv2.resize(mask_aug, (nw, nh))
            
            placed = False
            for _ in range(20):
                x_off = random.randint(0, w_bg - nw)
                y_off = random.randint(0, h_bg - nh)
                roi_mask = occupied_mask[y_off:y_off+nh, x_off:x_off+nw]
                if np.sum(cv2.bitwise_and(roi_mask, mask_aug)) == 0:
                    placed = True
                    break
            
            if not placed: continue
            
            roi_bg = bg[y_off:y_off+nh, x_off:x_off+nw]
            mask_f = cv2.merge([mask_aug.astype(float)/255.0]*3)
            bg[y_off:y_off+nh, x_off:x_off+nw] = (img_aug * mask_f + roi_bg * (1 - mask_f)).astype(np.uint8)
            occupied_mask[y_off:y_off+nh, x_off:x_off+nw] = cv2.bitwise_or(roi_mask, mask_aug)
            
            # Synthetic labels (also safely clipped)
            xc = (x_off+nw/2)/w_bg
            yc = (y_off+nh/2)/h_bg
            wn = nw/w_bg
            hn = nh/h_bg
            
            labels.append(f"{obj['cls']} {np.clip(xc,0,1):.6f} {np.clip(yc,0,1):.6f} {np.clip(wn,0,1):.6f} {np.clip(hn,0,1):.6f}")
            
        fname = f"collage_{i:05d}"
        cv2.imwrite(os.path.join(dest_root, 'images/train', fname+'.jpg'), bg)
        if labels:
            with open(os.path.join(dest_root, 'labels/train', fname+'.txt'), 'w') as f:
                f.write('\n'.join(labels))

    print(f"\n✅ Dataset generated successfully!")
    print(f"   [TRAIN] Real images (class 2): {stats['train_02_real']}")
    print(f"   [TRAIN] Objects cached for synth: {stats['train_other_cached']}")
    print(f"   [TRAIN] Synthetic collages: {num_collages}")
    print(f"   [TEST]  Real images (20% split): {stats['test_real']}")

    # Create YAML configuration
    # Note: We point 'val' to 'test' so training scripts have a validation set to use
    with open('./linemod.yaml', 'w') as f:
        f.write(f"path: {dest_root}\ntrain: images/train\nval: images/test\ntest: images/test\nnc: {len(class_names)}\nnames: {class_names}")
        
    return train_subset_all, test_subset_all

def create_student_dataset_final(dest_root, model_path, train_data_input, collage_root, conf_threshold=0.8, iou_threshold=0.45):
    """
    Step 3: Creates the Final Student Dataset.
    FIXED: optimized for dictionary input (from Step 1) and fixed YAML generation.
    """

    # --- 1. SETUP DIRECTORIES ---
    print(f"\n🚀 STEP 3: Creating Student Dataset in: {dest_root}")
    if os.path.exists(dest_root): shutil.rmtree(dest_root)
    
    dest_img_dir = os.path.join(dest_root, 'images', 'train')
    dest_lbl_dir = os.path.join(dest_root, 'labels', 'train')
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)

    print(f"🧠 Loading Teacher Model: {model_path}")
    model = YOLO(model_path)
    
    # --- HELPER: IoU Calculation ---
    def compute_iou(box1, box2):
        b1_x1, b1_y1 = box1[0]-box1[2]/2, box1[1]-box1[3]/2
        b1_x2, b1_y2 = box1[0]+box1[2]/2, box1[1]+box1[3]/2
        b2_x1, b2_y1 = box2[0]-box2[2]/2, box2[1]-box2[3]/2
        b2_x2, b2_y2 = box2[0]+box2[2]/2, box2[1]+box2[3]/2
        
        inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
        inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        union = b1_area + b2_area - inter_area
        return inter_area / union if union > 0 else 0

    stats = {'real': 0, 'pseudo': 0, 'collage': 0}

    # =========================================================
    # PART A: PROCESS REAL DATA (Optimized for Dictionary Input)
    # =========================================================
    print(f"📋 Processing {len(train_data_input)} real training elements...")
    
    for item in tqdm(train_data_input, desc="Pseudo-Labeling"):
        
        # 1. EXTRACT DATA FROM DICTIONARY
        # We assume input is the 'train_subset' list of dicts from Step 1
        if isinstance(item, dict):
            src_path = item['src']
            # Get labels directly from memory (faster/safer)
            curr_labels = list(item['labels']) 
            
            # Reconstruct unique filename
            # e.g., path .../01/rgb/0000.png -> folder="01"
            folder_str = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
            fname = f"real_{folder_str}_{item['fname']}"
        else:
            # Fallback for strings (just in case)
            src_path = item
            if not os.path.exists(src_path): continue
            fname = os.path.basename(src_path)
            
            # Read labels from disk
            lbl_path = src_path.replace('images', 'labels').replace('.png', '.txt')
            curr_labels = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    curr_labels = [line.strip() for line in f.readlines()]

        # 2. LOAD IMAGE
        img = cv2.imread(src_path)
        if img is None: continue

        # 3. PARSE GT BOXES (For duplicate checking)
        gt_boxes = [] 
        for l in curr_labels:
            try:
                parts = list(map(float, l.split()))
                if len(parts) == 5:
                    gt_boxes.append(parts) # [cls, xc, yc, w, h]
            except: pass

        # 4. RUN TEACHER (Pseudo-Labeling)
        # Skip target object folders (like Benchvise 02) to keep GT pure
        is_target_folder = "_02_" in fname or "/02/" in src_path
        
        if not is_target_folder:
            results = model.predict(img, conf=conf_threshold, verbose=False, iou=0.45)
            
            for r in results:
                for box in r.boxes:
                    p_cls = int(box.cls[0])
                    p_box = box.xywhn[0].tolist()
                    
                    # Duplicate Check
                    is_dup = False
                    for gt in gt_boxes:
                        if int(gt[0]) == p_cls and compute_iou(p_box, gt[1:]) > iou_threshold:
                            is_dup = True
                            break
                    
                    # Add new label if unique
                    if not is_dup:
                        curr_labels.append(f"{p_cls} {p_box[0]:.6f} {p_box[1]:.6f} {p_box[2]:.6f} {p_box[3]:.6f}")
                        stats['pseudo'] += 1
        
        # 5. SAVE TO DISK
        cv2.imwrite(os.path.join(dest_img_dir, fname), img)
        with open(os.path.join(dest_lbl_dir, fname.replace('.png','.txt')), 'w') as f:
            f.write('\n'.join(curr_labels))
            
        stats['real'] += 1

    '''
    # =========================================================
    # PART B: MERGE SYNTHETIC COLLAGES
    # =========================================================
    print("🎨 Merging Synthetic Collages...")
    src_synth_img = os.path.join(collage_root, 'images', 'train')
    src_synth_lbl = os.path.join(collage_root, 'labels', 'train')
    
    if os.path.exists(src_synth_img):
        synth_files = [f for f in os.listdir(src_synth_img) if f.endswith('.jpg') or f.endswith('.png')]
        for fn in tqdm(synth_files, desc="Copying Collages"):
            shutil.copy(os.path.join(src_synth_img, fn), os.path.join(dest_img_dir, fn))
            lbl_name = fn.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(os.path.join(src_synth_lbl, lbl_name)):
                shutil.copy(os.path.join(src_synth_lbl, lbl_name), os.path.join(dest_lbl_dir, lbl_name))
            stats['collage'] += 1
    '''

    # =========================================================
    # PART C: YAML CREATION (FIXED)
    # =========================================================
    # FIX: Use the standard path for the real dataset validation set.
    # This avoids crashes when trying to read paths from the dictionary.
    real_test_path = "/kaggle/working/datasets/images/test"
    
    class_names = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
    
    yaml_content = f"""
path: {dest_root}
train: images/train
val: {real_test_path}
test: {real_test_path}
nc: {len(class_names)}
names: {class_names}
"""
    with open(os.path.join(dest_root, 'student.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"\n✅ Student Dataset Completed!")
    print(f"   🖼️  Real Processed: {stats['real']}")
    print(f"   🤖 Pseudo-Labels Added: {stats['pseudo']}")
    #print(f"   🎨 Collages Merged: {stats['collage']}")
    print(f"   📄 Config: {os.path.join(dest_root, 'student.yaml')}")