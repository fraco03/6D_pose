from pandas import DataFrame
import torch
from src.model_evaluation import evalutation_pipeline
from utils.linemod_config import get_linemod_config
from utils.load_data import load_model_data, load_model
from src.pose_pointnet.model import PointNetPoseModel
from src.pose_pointnet.dataset import PointNetLineModDataset
from torch.utils.data import DataLoader

import numpy as np
import os
import pandas as pd
import open3d as o3d
import trimesh
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader

# Import dei tuoi moduli esistenti
from src.pose_pointnet.model import PointNetPoseModel
from src.pose_pointnet.dataset import PointNetLineModDataset
from src.model_evaluation import compute_ADD_metric_quaternion, compute_ADDs_metric_quaternion

def evaluate_POINTNET(
        model_path: str,
        dataset_root: str,
        output_path: str
) -> DataFrame:
    MODEL_PATH = model_path
    DATASET_ROOT = dataset_root
    OUTPUT_PATH = output_path
    config = get_linemod_config(DATASET_ROOT)

    print("📥 Loading 3D model points and diameters...")
    model_points, diameters = config.load_all_models_3d('mm')

    print("📦 Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        checkpoint_location=MODEL_PATH,
        device=device,
        model_class=PointNetPoseModel,
        num_points=1024
    )

    print("📚 Preparing test dataset and dataloader...")
    test_dataset = PointNetLineModDataset(
        root_dir=DATASET_ROOT,
        split="test",
        verbose=False
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    VALID_OBJ_IDS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15] 
    max_obj_id = max(VALID_OBJ_IDS)
        
        
    def prediction_function(model, batch, device):
        points = batch['points'].to(device)
        centroids = batch['centroid'].to(device)

        pred_q, pred_t_residual = model(points)

        pred_t = pred_t_residual + centroids

        return pred_q, pred_t

    def ground_truth_function(batch, device):
        gt_quats = batch['rotation'].to(device)
        gt_trans = batch['gt_translation'].to(device)

        return gt_quats, gt_trans

    df = evalutation_pipeline(
        model, 
        test_loader,
        device,
        ground_truth_function,
        prediction_function,
        model_points,
        diameters,
        report_file_path=OUTPUT_PATH,
        um='mm'
    )

    return df


# ICP helper function
def _refine_with_icp(pred_q, pred_t, source_points, target_points):
    """
    Executes the ICP refinement between the prediction and the observed points.
    """
    # 1. Convert Quaternion -> Matrix (Handles Scipy's x,y,z,w format)
    # Assume PointNet outputs (w, x, y, z). Scipy wants (x, y, z, w).
    # We do a roll just in case, if the error is high try removing it.
    try:
        # q_scipy = np.roll(pred_q, -1) # w,x,y,z -> x,y,z,w
        # If you use standard utilities it's usually handled, here we use raw scipy:
        rot_mat = R.from_quat(pred_q).as_matrix() # try first without roll
    except ValueError:
        rot_mat = np.eye(3)

    initial_transform = np.eye(4)
    initial_transform[:3, :3] = rot_mat
    initial_transform[:3, 3] = pred_t

    # 2. Setup Open3D Point Clouds
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points) # CAD Model
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points) # Camera Depth
    
    # 3. ICP
    # Threshold 2cm: quite large to capture, tight enough to not be too wrong
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, 0.02, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )
    
    # 4. Extract refined pose
    refined_T = reg_p2p.transformation
    refined_q = R.from_matrix(refined_T[:3, :3]).as_quat()
    refined_t = refined_T[:3, 3]
    
    return refined_q, refined_t




# evaluation function with ICP refinement
def evaluate_POINTNET_ICP(model_path, dataset_root, output_path, device="cuda"):
    """
    Valuta il modello PointNet E applica ICP per raffinare la posa.
    Genera un report che confronta le prestazioni prima e dopo.
    """
    print(f"🚀 Starting Evaluation with ICP Refinement...")
    print(f"📂 Model: {model_path}")
    print(f"📂 Dataset: {dataset_root}")
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 1. Carica il Dataset (Split TEST)
    test_dataset = PointNetLineModDataset(root_dir=dataset_root, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 per semplicità nell'ICP
    
    # 2. Carica il Modello
    model = PointNetPoseModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Gestione DataParallel se il modello è stato salvato con 'module.'
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Pre-carica i Modelli 3D (Mesh) per l'ICP
    # Serve per avere la nuvola "Sorgente" pulita (Canonical)
    print("⏳ Loading 3D CAD Models for ICP...")
    mesh_cache = {}
    diameters = {}
    models_dir = os.path.join(dataset_root, "models")
    
    id_to_name = test_dataset.id_to_class
    valid_ids = test_dataset.VALID_OBJECTS
    
    for oid in valid_ids:
        try:
            ply_path = os.path.join(models_dir, f"obj_{oid:02d}.ply")
            mesh = trimesh.load(ply_path)
            # Sample 1024 punti per l'ICP e per la metrica ADD
            pts = trimesh.sample.sample_surface(mesh, 1024)[0] / 1000.0 # Converti in Metri
            mesh_cache[oid] = pts
            diameters[oid] = np.linalg.norm(mesh.extents / 1000.0)
        except Exception as e:
            print(f"⚠️ Warning: Could not load model for ID {oid}: {e}")

    # 4. Loop di Valutazione
    results = []
    # Oggetti simmetrici (usa ADD-S)
    SYMMETRIC_IDS = [10, 11] # Eggbox, Glue
    
    print("running evaluation loop...")
    for batch in tqdm(test_loader):
        points = batch['points'].to(device)      # Input (Centered)
        centroids = batch['centroid'].to(device) # Centroid
        gt_q = batch['rotation'].numpy()         # GT Quat
        gt_t = batch['gt_translation'].numpy()   # GT Trans (Absolute)
        obj_id = int(batch['object_id'])
        
        if obj_id not in mesh_cache: continue

        # --- A. PointNet Prediction ---
        with torch.no_grad():
            pred_q_raw, pred_t_res = model(points)
            
        pred_q_np = pred_q_raw.cpu().numpy()[0]
        # Traslazione Assoluta = Centroide + Residuo
        pred_t_abs_np = (centroids + pred_t_res).cpu().numpy()[0]
        
        # --- B. ICP Refinement ---
        # Target Points: Input points (centrati) + Centroide reale = Punti nello spazio camera
        # Nota: batch['points'] è (B, 3, N), lo vogliamo (N, 3)
        input_pts_centered = points[0].transpose(0, 1).cpu().numpy()
        target_pts_cam = input_pts_centered + centroids.cpu().numpy()[0]
        
        refined_q, refined_t = _refine_with_icp(
            pred_q_np, pred_t_abs_np,
            mesh_cache[obj_id], # Source (Model)
            target_pts_cam      # Target (Camera)
        )
        
        # --- C. Calcolo Metriche ---
        # Scegli ADD o ADD-S
        metric_func = compute_ADDs_metric_quaternion if obj_id in SYMMETRIC_IDS else compute_ADD_metric_quaternion
        
        # 1. Errore PointNet
        err_pn = metric_func(
            mesh_cache[obj_id], gt_q[0], gt_t[0], pred_q_np, pred_t_abs_np
        )
        
        # 2. Errore ICP
        err_icp = metric_func(
            mesh_cache[obj_id], gt_q[0], gt_t[0], refined_q, refined_t
        )
        
        results.append({
            "Object_ID": obj_id,
            "Name": id_to_name[obj_id],
            "Diameter_m": diameters[obj_id],
            "PN_Error_m": err_pn,
            "ICP_Error_m": err_icp,
            "Improvement_m": err_pn - err_icp
        })

    # 5. Aggregazione Risultati
    df = pd.DataFrame(results)
    
    # Calcola statistiche per oggetto
    summary = []
    for oid in valid_ids:
        subset = df[df["Object_ID"] == oid]
        if subset.empty: continue
        
        diam = diameters[oid]
        thresh = diam * 0.1
        
        # PointNet Stats
        pn_mean = subset["PN_Error_m"].mean()
        pn_acc = (subset["PN_Error_m"] < thresh).mean() * 100
        
        # ICP Stats
        icp_mean = subset["ICP_Error_m"].mean()
        icp_acc = (subset["ICP_Error_m"] < thresh).mean() * 100
        
        summary.append({
            "Name": id_to_name[oid],
            "Diam(cm)": round(diam*100, 1),
            "PN Err(cm)": round(pn_mean*100, 2),
            "ICP Err(cm)": round(icp_mean*100, 2),
            "Diff(cm)": round((pn_mean - icp_mean)*100, 2),
            "PN Acc(%)": round(pn_acc, 1),
            "ICP Acc(%)": round(icp_acc, 1)
        })
        
    summary_df = pd.DataFrame(summary)
    
    # Global Average
    print("\n" + "="*80)
    print(" EVALUATION REPORT: POINTNET vs ICP")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("-" * 80)
    print(f"Global PN Error:  {df['PN_Error_m'].mean()*100:.2f} cm")
    print(f"Global ICP Error: {df['ICP_Error_m'].mean()*100:.2f} cm")
    print("="*80)
    
    # Salva
    summary_df.to_csv(output_path, index=False)
    print(f"✅ Report saved to {output_path}")