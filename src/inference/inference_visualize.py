import cv2
import numpy as np
from src.pose_rgb.pose_utils import quaternion_to_rotation_matrix

# --- FUNZIONI DI SUPPORTO MATEMATICO ---

def project_3d_points(points_3d, K, R_quat, t_vec):
    """
    Proietta punti 3D generici sull'immagine 2D.
    Nessuna dipendenza da dataset.
    
    Args:
        points_3d: (N, 3) numpy array (in Metri).
        K: (3, 3) numpy array (Matrice Intrinseca).
        R_quat: (4,) numpy array [w, x, y, z] o [x, y, z, w].
        t_vec: (3,) numpy array [x, y, z] (in Metri).
    """
    if points_3d.shape[1] != 3:
        raise ValueError(f"points_3d shape deve essere (N, 3), ottenuto {points_3d.shape}")

    # Gestione formato Quaternione (Assumiamo standard [w, x, y, z] se non specificato)
    # Se il tuo quaternion_to_rotation_matrix si aspetta [x, y, z, w], adatta qui.
    R = quaternion_to_rotation_matrix(R_quat) 
    
    t_vec = t_vec.reshape(3, 1)

    # Trasformazione World -> Camera
    points_cam = (R @ points_3d.T) + t_vec # (3, N)
    
    # Proiezione Pinhole
    points_2d_hom = K @ points_cam # (3, N)
    
    # Normalizzazione omogenea (u = x/z, v = y/z)
    points_2d = points_2d_hom[:2, :] / (points_2d_hom[2, :] + 1e-8)
    
    return points_2d.T # (N, 2)

# --- FUNZIONI DI DISEGNO ---

def draw_3d_bbox_on_image(image, points_2d, color=(255, 0, 255), thickness=2):
    """
    Disegna il wireframe del parallelepipedo usando gli 8 corner proiettati.
    L'ordine dei punti deve essere quello standard (min_z[4], max_z[4]).
    """
    img_vis = image.copy()
    points = [tuple(pt.astype(int)) for pt in points_2d]
    
    # Definizione connessioni (Edges)
    # Base inferiore (0-3), Base superiore (4-7), Pilastri (0-4, etc.)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Base Inferiore
        (4, 5), (5, 6), (6, 7), (7, 4), # Base Superiore
        (0, 4), (1, 5), (2, 6), (3, 7)  # Verticali
    ]
    
    for s, e in edges:
        cv2.line(img_vis, points[s], points[e], color, thickness)
        
    return img_vis

def draw_3d_axes_on_image(image, K, R_quat, t_vec, axis_len=0.08, thickness=2):
    """
    Disegna gli assi XYZ al centro dell'oggetto.
    Crea i punti 3D degli assi al volo e li proietta.
    """
    img_vis = image.copy()
    
    # Definizione assi nello spazio locale oggetto
    origin = np.array([0, 0, 0], dtype=np.float32)
    axis_x = np.array([axis_len, 0, 0], dtype=np.float32)
    axis_y = np.array([0, axis_len, 0], dtype=np.float32)
    axis_z = np.array([0, 0, axis_len], dtype=np.float32)
    
    points_3d = np.vstack([origin, axis_x, axis_y, axis_z]) # (4, 3)
    
    # Proiezione
    points_2d = project_3d_points(points_3d, K, R_quat, t_vec)
    pts = [tuple(pt.astype(int)) for pt in points_2d]
    
    origin_pt = pts[0]
    
    # Disegno (BGR colors)
    cv2.line(img_vis, origin_pt, pts[1], (0, 0, 255), thickness) # X = Rosso (BGR: Blue-Green-Red -> 2)
    cv2.line(img_vis, origin_pt, pts[2], (0, 255, 0), thickness) # Y = Verde
    cv2.line(img_vis, origin_pt, pts[3], (255, 0, 0), thickness) # Z = Blu
    
    return img_vis

# --- FUNZIONE HELPER DI INTEGRAZIONE ---

import random

def compute_and_draw_prediction(
    image, 
    model_bbox_3d, # (8, 3) array dei corner del modello in metri
    K, 
    pred_quat, 
    pred_trans,
    conf_score=None,
    label=None
):
    """
    Wrapper one-shot per fare tutto: proiezione e disegno bbox + assi.
    """
    # 1. Proietta BBox
    bbox_2d = project_3d_points(model_bbox_3d, K, pred_quat, pred_trans)
    
    # 2. Disegna BBox
    (r, g, b) = random.choices(range(256), k=3)
    res_img = draw_3d_bbox_on_image(image, bbox_2d, color=(r, g, b), thickness=2)
    
    # 3. Disegna Assi
    # res_img = draw_3d_axes_on_image(res_img, K, pred_quat, pred_trans)
    
    # 4. (Opzionale) Testo
    if label or conf_score:
        # txt = f"{label if label else ''} {conf_score:.2f}" if conf_score else label
        txt = label if label else ""
        # Trova il punto più in alto a sinistra della bbox proiettata per il testo
        txt_pos = tuple(np.min(bbox_2d, axis=0).astype(int))
        cv2.putText(res_img, txt, (txt_pos[0], txt_pos[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    return res_img