from pandas import DataFrame
import torch
from src.model_evaluation import evalutation_pipeline
from utils.linemod_config import get_linemod_config
from utils.load_data import load_model_data, load_model
from src.pose_pointnet.model import PointNetPoseModel
from src.pose_pointnet.dataset import PointNetLineModDataset
from torch.utils.data import DataLoader

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