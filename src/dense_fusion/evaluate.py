from pandas import DataFrame
import torch
from src.model_evaluation import evalutation_pipeline
from utils.linemod_config import get_linemod_config
from utils.load_data import load_model_data, load_model
from src.dense_fusion.model import FusionPoseModel         
from src.dense_fusion.dataset import DenseFusionLineModDataset 
from torch.utils.data import DataLoader
import os

def evaluate_DENSEFUSION(
        model_path: str,
        dataset_root: str,
        output_path: str,
        yolo_path: str
) -> DataFrame:
    MODEL_PATH = model_path
    DATASET_ROOT = dataset_root
    OUTPUT_PATH = output_path
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Errore: Il file del modello non esiste in: {MODEL_PATH}\nControlla la variabile 'checkpoint_path' del training precedente.")

    config = get_linemod_config(DATASET_ROOT)

    print("📥 Loading 3D model points and diameters...")
    model_points, diameters = config.load_all_models_3d('mm')

    print(f"📦 Loading trained DenseFusion model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- LOAD MODEL DENSEFUSION ---
    model = load_model(
        checkpoint_location=MODEL_PATH,
        device=device,
        model_class=FusionPoseModel,
        num_points=1024
    )

    print("📚 Preparing test dataset and dataloader...")
    # --- LOAD DATASET ---
    test_dataset = DenseFusionLineModDataset(
        root_dir=DATASET_ROOT,
        split="test",
        verbose=False,
        yolo_path=yolo_path
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    def prediction_function(model, batch, device):
        # Unwrap input data
        points = batch['points'].to(device)
        images = batch['rgb'].to(device)   # RGB Images
        centroids = batch['centroid'].to(device)

        # Forward pass
        pred_q, pred_t_residual = model(points, images)

        # Absolute translation
        pred_t = pred_t_residual + centroids

        return pred_q, pred_t

    def ground_truth_function(batch, device):
        gt_quats = batch['rotation'].to(device)
        gt_trans = batch['gt_translation'].to(device)
        return gt_quats, gt_trans

    # Standard evaluation pipeline
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