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

# CHANGE iou FOR DIFFERENET mAP METRICS
def calculate_adapted_map50(model_path, source_root, iou_thresh=0.5):
    """
    Calculate adapted mAP@50 for the partially labeled LineMOD dataset.
    
    Strategy:
    For each image, only predictions of the class belonging to the specific folder
    are considered valid. Other predictions are masked out (ignored), so they don't
    count as False Positives when labels are missing.
    
    Args:
        model_path (str): Path to trained YOLO model weights
        source_root (str): Root directory of LineMOD dataset
        iou_thresh (float): IoU threshold for considering a detection correct (default 0.5)
        
    Returns:
        float: Mean Average Precision at IoU threshold
    """
    # Configuration
    valid_obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    class_names = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 
                   'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
    id_map = {obj_id: i for i, obj_id in enumerate(valid_obj_ids)}
    obj_folders = [f"{i:02d}" for i in valid_obj_ids]

    print(f"⚖️  Computing adapted mAP@{int(iou_thresh*100)} (masked evaluation)...")
    model = YOLO(model_path)
    
    # Data structures for AP calculation per class
    class_preds = {i: [] for i in range(len(class_names))}  # [confidence, is_true_positive]
    class_total_gt = {i: 0 for i in range(len(class_names))}  # Total ground truths per class

    # Main evaluation loop
    for folder_str in tqdm(obj_folders, desc="Evaluating"):
        base_dir = os.path.join(source_root, folder_str)
        rgb_dir = os.path.join(base_dir, 'rgb')
        gt_path = os.path.join(base_dir, 'gt.yml')
        train_txt = os.path.join(base_dir, 'train.txt')
        
        if not os.path.exists(gt_path): 
            continue

        # Load training IDs to exclude them from evaluation
        train_ids = set()
        if os.path.exists(train_txt):
            with open(train_txt, 'r') as f:
                train_ids = {line.strip() for line in f.readlines()}
        
        with open(gt_path, 'r') as f: 
            gt_data = yaml.safe_load(f)
        
        target_cls_idx = id_map[int(folder_str)]

        for img_id_int, anns in gt_data.items():
            img_id_str = str(img_id_int)
            if img_id_str in train_ids: 
                continue  # Skip training images
            
            fname = f"{img_id_int:04d}.png"
            img_path = os.path.join(rgb_dir, fname)
            if not os.path.exists(img_path): 
                continue
            
            # Collect ground truth boxes for target class only
            gt_boxes = []
            for ann in anns:
                oid = int(ann['obj_id'])
                if oid == int(folder_str):
                    x, y, w, h = ann['obj_bb']
                    gt_boxes.append([x, y, x+w, y+h])  # Convert to [x1, y1, x2, y2]
            
            if not gt_boxes: 
                continue
            
            class_total_gt[target_cls_idx] += len(gt_boxes)
            
            # Run YOLO prediction with low confidence threshold for full PR curve
            results = model.predict(img_path, conf=0.01, verbose=False)
            
            # Filter predictions: keep only target class predictions
            valid_preds = []
            for box in results[0].boxes:
                cls_pred = int(box.cls[0])
                if cls_pred == target_cls_idx:
                    x, y, w, h = box.xywh[0].tolist()
                    x1, y1 = x - w/2, y - h/2  # Convert center to corners
                    x2, y2 = x + w/2, y + h/2
                    conf = float(box.conf[0])
                    valid_preds.append({'bbox': [x1, y1, x2, y2], 'conf': conf})
            
            # Match predictions to ground truth (sorted by confidence descending)
            valid_preds.sort(key=lambda x: x['conf'], reverse=True)
            gt_matched = [False] * len(gt_boxes)
            
            for pred in valid_preds:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for idx, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(pred['bbox'], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                
                # Classify as TP or FP
                if best_iou >= iou_thresh:
                    if not gt_matched[best_gt_idx]:
                        class_preds[target_cls_idx].append([pred['conf'], 1])  # True Positive
                        gt_matched[best_gt_idx] = True
                    else:
                        class_preds[target_cls_idx].append([pred['conf'], 0])  # FP (duplicate)
                else:
                    class_preds[target_cls_idx].append([pred['conf'], 0])  # FP (low IoU)

    # --- CALCOLO AP (AVERAGE PRECISION) PER CLASSE ---
    print("\n" + "="*50)
    print(f"{'CLASS':<15} | {'AP@50':<10} | {'GT Count':<10}")
    print("-" * 50)
    

    print("\n" + "="*50)
    print(f"{'CLASS':<15} | {'AP@50':<10} | {'GT Count':<10}")
    print("-" * 50)
    
    aps = []
    
    for cls_idx in range(len(class_names)):
        preds = class_preds[cls_idx]
        total_gt = class_total_gt[cls_idx]
        
        if total_gt == 0:
            continue
            
        if not preds:
            ap = 0.0
        else:
            preds = np.array(preds)
            tp = np.cumsum(preds[:, 1])
            fp = np.cumsum(1 - preds[:, 1])
            
            rec = tp / total_gt
            prec = tp / (tp + fp + 1e-16)
            
            # Compute AP using area under precision-recall curve
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            
            # Make precision monotonically decreasing
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
                
            # Calculate area under curve
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            
        aps.append(ap)
        print(f"{class_names[cls_idx]:<15} | {ap:.4f}     | {total_gt}")
    
    mean_ap = np.mean(aps) if aps else 0.0
    print("-" * 50)
    print(f"Adapted mAP@50  | {mean_ap:.4f}")
    print("="*50)
    
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

def generate_synthetic_dataset(source_root, dest_root, bg_cache_dir, num_collages, max_objects_images):
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

            # Prepare YOLO labels
            yolo_labels = []
            for ann in anns:
                oid = int(ann['obj_id'])
                if oid in id_map:
                    img_w, img_h = 640, 480
                    x, y, w, h = ann['obj_bb']
                    xc, yc = (x + w/2)/img_w, (y + h/2)/img_h
                    wn, hn = w/img_w, h/img_h
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
            collage_samples = random.sample(train_subset, min(len(train_subset), 50))
            
            for img_data in collage_samples:
                if os.path.exists(img_data['mask']):
                    img = cv2.imread(img_data['src'])
                    mask = cv2.imread(img_data['mask'], cv2.IMREAD_GRAYSCALE)
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
            
            labels.append(f"{obj['cls']} {(x_off+nw/2)/w_bg:.6f} {(y_off+nh/2)/h_bg:.6f} {nw/w_bg:.6f} {nh/h_bg:.6f}")
            
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

def auto_labeled_dataset(dest_root, model_path, source_root, conf_threshold, iou_threshold):
    valid_obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    id_map = {obj_id: i for i, obj_id in enumerate(valid_obj_ids)}
    obj_folders = [f"{i:02d}" for i in valid_obj_ids]

    # --- SETUP DIRECTORIES ---
    if os.path.exists(dest_root): shutil.rmtree(dest_root)
    # Create 'train' and 'test' directories (80/20 split)
    for split in ['train', 'test']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(dest_root, dtype, split), exist_ok=True)

    print(f"🧠 Loading model: {model_path}")
    model = YOLO(model_path)

    # --- HELPER FUNCTIONS ---
    def convert_box_gt_to_yolo(size, box):
        """Convert bounding box from pixel coordinates [x, y, w, h] to YOLO format [xc, yc, w, h] normalized"""
        dw, dh = 1. / size[0], 1. / size[1]
        xc = (box[0] + box[2] / 2.0) * dw
        yc = (box[1] + box[3] / 2.0) * dh
        w = box[2] * dw
        h = box[3] * dh
        return [xc, yc, w, h]

    def compute_iou_yolo(box1, box2):
        """Calculate IoU between two boxes in YOLO format [xc, yc, w, h]"""
        def to_corners(b):
            return [b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2]
        
        b1 = to_corners(box1)
        b2 = to_corners(box2)
        
        inter_x1 = max(b1[0], b2[0])
        inter_y1 = max(b1[1], b2[1])
        inter_x2 = min(b1[2], b2[2])
        inter_y2 = min(b1[3], b2[3])
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        b1_area = (b1[2]-b1[0]) * (b1[3]-b1[1])
        b2_area = (b2[2]-b2[0]) * (b2[3]-b2[1])
        
        union = b1_area + b2_area - inter_area
        return inter_area / union if union > 0 else 0

    # --- MAIN LOOP ---
    print("🚀 Starting auto-labeling process with 80/20 split...")
    total_added = 0
    stats = {'train': 0, 'test': 0}

    for folder_str in obj_folders:
        folder_int = int(folder_str)
        base_dir = os.path.join(source_root, folder_str)
        rgb_dir = os.path.join(base_dir, 'rgb')
        gt_path = os.path.join(base_dir, 'gt.yml')
        
        if not os.path.exists(gt_path): continue
        
        # 1. Load all Ground Truth Data
        with open(gt_path, 'r') as f: gt_data = yaml.safe_load(f)
        all_img_ids = list(gt_data.keys())
        
        # 2. Shuffle and Split (80% Train, 20% Test)
        random.shuffle(all_img_ids)
        split_idx = int(len(all_img_ids) * 0.80)
        
        train_ids = set(all_img_ids[:split_idx])
        test_ids = set(all_img_ids[split_idx:])
        
        for img_key in tqdm(all_img_ids, desc=f"Processing Folder {folder_str}"):
            img_key_str = str(img_key)
            fname = f"{img_key:04d}.png"
            src_img_path = os.path.join(rgb_dir, fname)
            
            if not os.path.exists(src_img_path): continue
            
            # Determine subset
            if img_key in train_ids:
                subset = 'train'
                stats['train'] += 1
            else:
                subset = 'test'
                stats['test'] += 1
                
            img = cv2.imread(src_img_path)
            h_img, w_img = img.shape[:2]
            
            # Collect Ground Truth Labels
            final_labels = [] 
            gt_boxes_yolo = [] 
            
            annotations = gt_data.get(img_key, [])
            for ann in annotations:
                obj_id = int(ann['obj_id'])
                if obj_id in id_map:
                    cls_id = id_map[obj_id]
                    bbox_yolo = convert_box_gt_to_yolo((w_img, h_img), ann['obj_bb'])
                    
                    gt_boxes_yolo.append([cls_id] + bbox_yolo)
                    final_labels.append(f"{cls_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}")
            
            # --- AUTO-LABELING LOGIC ---
            # 1. Only run on TRAINING data (Test data must remain Ground Truth only)
            # 2. Skip Folder 02 (Benchvise) completely for auto-labeling as requested previously
            if subset == 'train' and folder_int != 2:
                results = model.predict(img, conf=conf_threshold, verbose=False, iou=0.4)
                
                for r in results:
                    for box in r.boxes:
                        p_cls = int(box.cls[0])
                        p_xywh = box.xywhn[0].tolist()
                        
                        # Check against Ground Truth to avoid duplicates
                        is_duplicate = False
                        for gt_b in gt_boxes_yolo:
                            gt_cls = gt_b[0]
                            gt_geom = gt_b[1:]
                            
                            if p_cls == gt_cls:
                                iou = compute_iou_yolo(p_xywh, gt_geom)
                                if iou > iou_threshold:
                                    is_duplicate = True
                                    break
                        
                        # Add new label if unique
                        if not is_duplicate:
                            final_labels.append(f"{p_cls} {p_xywh[0]:.6f} {p_xywh[1]:.6f} {p_xywh[2]:.6f} {p_xywh[3]:.6f}")
                            total_added += 1

            # Save image and labels
            dst_fname = f"{folder_str}_{fname}"
            cv2.imwrite(os.path.join(dest_root, 'images', subset, dst_fname), img)
            
            if final_labels:
                with open(os.path.join(dest_root, 'labels', subset, dst_fname.replace('.png', '.txt')), 'w') as f:
                    f.write('\n'.join(final_labels))

    print(f"\n✅ Dataset generated successfully!")
    print(f"   [TRAIN] Images processed: {stats['train']}")
    print(f"   [TEST]  Images processed: {stats['test']}")
    print(f"   ➕ Extra labels added automatically (Train only): {total_added}")
    print(f"   📁 Saved to: {dest_root}")

    # Create YAML configuration
    # Note: 'val' points to 'test' so training scripts have a valid validation path
    names = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
    with open(os.path.join(dest_root, 'data.yaml'), 'w') as f:
        f.write(f"path: {dest_root}\ntrain: images/train\nval: images/test\ntest: images/test\nnc: 13\nnames: {names}")