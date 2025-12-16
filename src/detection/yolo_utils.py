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

    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(dest_root, dtype, split), exist_ok=True)

    # Download random background images
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
        """Rotate image by given angle and adjust canvas size to fit"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    def get_train_ids(folder_path):
        """Read train.txt and return set of training image IDs"""
        txt_path = os.path.join(folder_path, 'train.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                return {line.strip() for line in f.readlines()}
        return set()

    # --- PREPARE COLLAGE OBJECTS (EXCLUDING CLASS 02) ---
    print("♻️  Loading objects for collages (excluding class 02)...")
    objects_cache = []

    for folder_str in tqdm(obj_folders):
        # Skip class 02 - use real images for this class instead of collages
        if int(folder_str) == 2:
            continue

        class_id = id_map[int(folder_str)]
        base_dir = os.path.join(source_root, folder_str)
        
        # Load only training images to avoid data leakage
        train_ids = get_train_ids(base_dir)
        if not train_ids: continue

        # Sample up to 50 images to avoid memory issues
        valid_files = [f"{int(x):04d}.png" for x in train_ids]
        sampled_files = random.sample(valid_files, min(len(valid_files), 50))

        for fname in sampled_files:
            img_path = os.path.join(base_dir, 'rgb', fname)
            mask_path = os.path.join(base_dir, 'mask', fname)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                x, y, w, h = cv2.boundingRect(mask)
                if w > 5 and h > 5:
                    crop_img = img[y:y+h, x:x+w]
                    crop_mask = mask[y:y+h, x:x+w]
                    _, crop_mask = cv2.threshold(crop_mask, 127, 255, cv2.THRESH_BINARY)
                    objects_cache.append({'cls': class_id, 'img': crop_img, 'mask': crop_mask})

    if not objects_cache: raise RuntimeError("❌ Object cache is empty! Check paths.")

    # --- GENERATE SYNTHETIC COLLAGES (ALL FOR TRAINING) ---
    print(f"🚀 Generating {num_collages} synthetic collages...")

    for i in tqdm(range(num_collages)):
        bg = cv2.imread(random.choice(bg_files))
        bg = cv2.resize(bg, (640, 640))
        h_bg, w_bg = bg.shape[:2]
        
        labels = []
        occupied_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
        
        for _ in range(random.randint(1, max_objects_images)):
            obj = random.choice(objects_cache)
            img_aug, mask_aug = obj['img'].copy(), obj['mask'].copy()
            
            # Apply rotation and scaling augmentation
            angle = random.randint(-180, 180)
            img_aug = rotate_image(img_aug, angle)
            mask_aug = rotate_image(mask_aug, angle)
            
            scale = random.uniform(0.4, 0.9)
            nh, nw = int(img_aug.shape[0] * scale), int(img_aug.shape[1] * scale)
            if nh <= 0 or nw <= 0: continue
            img_aug = cv2.resize(img_aug, (nw, nh))
            mask_aug = cv2.resize(mask_aug, (nw, nh))
            
            # Find non-overlapping position
            placed = False
            for _ in range(20):
                x_off = random.randint(0, w_bg - nw)
                y_off = random.randint(0, h_bg - nh)
                roi_mask = occupied_mask[y_off:y_off+nh, x_off:x_off+nw]
                if np.sum(cv2.bitwise_and(roi_mask, mask_aug)) == 0:
                    placed = True
                    break
            
            if not placed: continue
            
            # Paste object onto background using alpha blending
            roi_bg = bg[y_off:y_off+nh, x_off:x_off+nw]
            mask_f = cv2.merge([mask_aug.astype(float)/255.0]*3)
            bg[y_off:y_off+nh, x_off:x_off+nw] = (img_aug * mask_f + roi_bg * (1 - mask_f)).astype(np.uint8)
            occupied_mask[y_off:y_off+nh, x_off:x_off+nw] = cv2.bitwise_or(roi_mask, mask_aug)
            
            labels.append(f"{obj['cls']} {(x_off+nw/2)/w_bg:.6f} {(y_off+nh/2)/h_bg:.6f} {nw/w_bg:.6f} {nh/h_bg:.6f}")
            
        # Save to training set
        fname = f"collage_{i:05d}"
        cv2.imwrite(os.path.join(dest_root, 'images/train', fname+'.jpg'), bg)
        if labels:
            with open(os.path.join(dest_root, 'labels/train', fname+'.txt'), 'w') as f:
                f.write('\n'.join(labels))

    # --- PROCESS REAL IMAGES (MIXED LOGIC BY CLASS) ---
    print("📸 Processing real images...")
    stats = {'train_02': 0, 'skipped_train_others': 0, 'val': 0}

    for folder_str in tqdm(obj_folders):
        is_class_02 = (int(folder_str) == 2)
        base_dir = os.path.join(source_root, folder_str)
        gt_path = os.path.join(base_dir, 'gt.yml')
        
        if not os.path.exists(gt_path): continue
        
        train_ids = get_train_ids(base_dir)
        with open(gt_path, 'r') as f: gt_data = yaml.safe_load(f)
        
        for img_id_int, anns in gt_data.items():
            img_id_str = str(img_id_int)
            fname = f"{img_id_int:04d}.png"
            src_img = os.path.join(base_dir, 'rgb', fname)
            
            if not os.path.exists(src_img): continue
            
            is_in_train_txt = (img_id_str in train_ids)
            
            if is_in_train_txt:
                # Training image
                if is_class_02:
                    # Use real images for class 02 in training
                    target_subset = 'train'
                    stats['train_02'] += 1
                else:
                    # Skip training images for other classes (use collages instead)
                    stats['skipped_train_others'] += 1
                    continue 
            else:
                # Validation image - use for all classes
                target_subset = 'val'
                stats['val'] += 1
            
            # Copy image and convert labels
            dst_fname = f"real_{folder_str}_{fname}"
            shutil.copy(src_img, os.path.join(dest_root, 'images', target_subset, dst_fname))
            
            yolo_labels = []
            for ann in anns:
                oid = int(ann['obj_id'])
                if oid in id_map:
                    img_w, img_h = 640, 480  # Standard LineMOD dimensions
                    x, y, w, h = ann['obj_bb']
                    xc, yc = (x + w/2)/img_w, (y + h/2)/img_h
                    wn, hn = w/img_w, h/img_h
                    yolo_labels.append(f"{id_map[oid]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                    
            if yolo_labels:
                with open(os.path.join(dest_root, 'labels', target_subset, dst_fname.replace('.png','.txt')), 'w') as f:
                    f.write('\n'.join(yolo_labels))

    print(f"\n✅ Dataset generated successfully!")
    print(f"   [TRAINING]  Synthetic collages (no class 2):  {num_collages}")
    print(f"   [TRAINING]  Real images (class 2 only):       {stats['train_02']}")
    print(f"   [SKIPPED]   Real training images (others):    {stats['skipped_train_others']}")
    print(f"   [VALIDATION] Real images (all classes):       {stats['val']}")

    # Create YAML configuration
    with open('./linemod.yaml', 'w') as f:
        f.write(f"path: {dest_root}\ntrain: images/train\nval: images/val\nnc: {len(class_names)}\nnames: {class_names}")
        

def auto_labeled_dataset(dest_root, model_path, source_root, conf_threshold, iou_threshold):
    valid_obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    id_map = {obj_id: i for i, obj_id in enumerate(valid_obj_ids)}
    obj_folders = [f"{i:02d}" for i in valid_obj_ids]

    # --- SETUP DIRECTORIES ---
    if os.path.exists(dest_root): shutil.rmtree(dest_root)
    for split in ['train', 'val']:
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

    def get_train_ids(folder_path):
        """Read train.txt and return set of training image IDs"""
        txt_path = os.path.join(folder_path, 'train.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                return {line.strip() for line in f.readlines()}
        return set()

    # --- MAIN LOOP ---
    print("🚀 Starting auto-labeling process...")
    total_added = 0

    for folder_str in obj_folders:
        folder_int = int(folder_str)
        base_dir = os.path.join(source_root, folder_str)
        
        train_ids_set = get_train_ids(base_dir)
        rgb_dir = os.path.join(base_dir, 'rgb')
        gt_path = os.path.join(base_dir, 'gt.yml')
        
        if not os.path.exists(gt_path): continue
        
        with open(gt_path, 'r') as f: gt_data = yaml.safe_load(f)
        img_ids = list(gt_data.keys())
        
        for img_key in tqdm(img_ids, desc=f"Folder {folder_str}"):
            img_key_str = str(img_key)
            fname = f"{img_key:04d}.png"
            src_img_path = os.path.join(rgb_dir, fname)
            
            if not os.path.exists(src_img_path): continue
            
            # Determine split based on train.txt
            subset = 'train' if img_key_str in train_ids_set else 'val'
                
            img = cv2.imread(src_img_path)
            h_img, w_img = img.shape[:2]
            
            # Collect ground truth labels
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
            
            # Auto-labeling: only for training images, excluding folder 02 (already perfect)
            if subset == 'train' and folder_int != 2:
                results = model.predict(img, conf=conf_threshold, verbose=False, iou=0.4)
                
                for r in results:
                    for box in r.boxes:
                        p_cls = int(box.cls[0])
                        p_xywh = box.xywhn[0].tolist()
                        
                        # Check if prediction duplicates existing ground truth
                        is_duplicate = False
                        for gt_b in gt_boxes_yolo:
                            gt_cls = gt_b[0]
                            gt_geom = gt_b[1:]
                            
                            if p_cls == gt_cls:
                                iou = compute_iou_yolo(p_xywh, gt_geom)
                                if iou > iou_threshold:
                                    is_duplicate = True
                                    break
                        
                        # Add new label if not duplicate
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
    print(f"   ➕ Extra labels added automatically: {total_added}")
    print(f"   📁 Saved to: {dest_root}")

    # Create YAML configuration
    names = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
    with open(os.path.join(dest_root, 'data.yaml'), 'w') as f:
        f.write(f"path: {dest_root}\ntrain: images/train\nval: images/val\nnc: 13\nnames: {names}")