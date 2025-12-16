import os
import yaml
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

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
    
    apCalculate Average Precision per class
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
    