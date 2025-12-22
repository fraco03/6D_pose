from pandas import DataFrame

from src.model_evaluation import evalutation_pipeline
from utils.linemod_config import get_linemod_config
from utils.load_data import load_model_data, load_model
from src.pose_rgb.model import ResNetRotation, TranslationNet
from torch.utils.data import DataLoader
from src.pose_rgb.dataset import LineModPoseDataset
import torch
from src.pose_rgb.pose_utils import inverse_pinhole_projection


def evaluate_RGB(
        rot_model_path: str,
        trans_model_path: str,
        dataset_root: str,
        output_path: str
) -> DataFrame:
    ROT_MODEL_PATH = rot_model_path
    TRANS_MODEL_PATH = trans_model_path
    DATASET_ROOT = dataset_root
    OUTPUT_PATH = output_path

    config = get_linemod_config(DATASET_ROOT)

    print("📥 Loading 3D model points and diameters...")
    model_points, diameters = config.load_all_models_3d('mm')

    print("📦 Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Rotation Model
    model_rot = load_model(ROT_MODEL_PATH, device, model_class=ResNetRotation, model_key='model_state_dict', freeze_backbone=False)
    model_trans = load_model(TRANS_MODEL_PATH, device, model_class=TranslationNet, model_key='model_state_dict')
    model = (model_rot, model_trans)
    

    print("📚 Preparing test dataset and dataloader...")
    test_dataset = LineModPoseDataset(
        root_dir=DATASET_ROOT,
        split="test",
        verbose=False
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)



    def prediction_function(model, batch, device):
        (model_rot, model_trans) = model
        imgs = batch['image'].to(device)
        bbox_info = batch['bbox_info'].to(device)
        bbox_center = batch['bbox_center'].to(device) # (B, 2) [cx_pix, cy_pix]
        cam_K = batch['cam_K'].to(device)             # (B, 3, 3)
        original_width = batch['original_width'].to(device)
        original_height = batch['original_height'].to(device)

        # Quaternion prediction
        pred_quats = model_rot(imgs)

        # Translation prediction pipeline
        preds_raw = model_trans(imgs, bbox_info)
        pred_relative_deltas = preds_raw[:, :2] 

        W_real = bbox_info[:, 2] * original_width
        H_real = bbox_info[:, 3] * original_height

        real_dims = torch.stack([W_real, H_real], dim=1)
        pred_deltas = pred_relative_deltas * real_dims

        pred_log_z = preds_raw[:, 2]   # (B, )  -> log(z)
        pred_z = torch.exp(pred_log_z).clamp(min=0.1, max=10.0)

        pred_trans = inverse_pinhole_projection(
            crop_center=bbox_center,
            deltas=pred_deltas,
            z=pred_z,
            cam_K=cam_K
        )

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


def evaluate_RGB_rot_only(
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
    model_data = load_model_data(MODEL_PATH, map_location=device, model_key='model_state_dict')
    model = ResNetRotation(freeze_backbone=False)
    model.load_state_dict(model_data['model_state_dict'])
    model = model.to(device)

    print("📚 Preparing test dataset and dataloader...")
    test_dataset = LineModPoseDataset(
        root_dir=DATASET_ROOT,
        split="test",
        verbose=False
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)



    def prediction_function(model, batch, device):
        imgs = batch['image'].to(device)
        pred_quats = model(imgs)
        pred_trans = batch['translation'].to(device)  # Use ground truth translation
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