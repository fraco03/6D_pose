from pandas import DataFrame

from src.model_evaluation import evalutation_pipeline
from utils.linemod_config import get_linemod_config
from utils.load_data import load_model_data
from src.pose_rgbd.model import RGBDRotationModel
from torch.utils.data import DataLoader
from src.pose_rgbd.dataset import LineModPoseDepthDataset
import torch

def evaluate_RGBD(
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
    model_data = load_model_data(MODEL_PATH, map_location=device)
    model = RGBDRotationModel()
    model.load_state_dict(model_data['model_state_dict'])
    model = model.to(device)


    print("📚 Preparing test dataset and dataloader...")
    test_dataset = LineModPoseDepthDataset(
        root_dir=DATASET_ROOT,
        split="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)



    def prediction_function(model, batch, device):
        imgs = batch['image'].to(device)
        depths = batch['depth'].to(device)
        pred_quats = model(imgs, depths)
        pred_trans = batch['3D_center'].to(device)
        return pred_quats, pred_trans

    def ground_truth_function(batch, device):
        gt_quats = batch['rotation'].to(device)
        gt_trans = batch['translation'].to(device)
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

        