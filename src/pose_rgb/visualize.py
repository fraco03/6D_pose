from utils.projection_utils import setup_projection_utils, visualize_random_samples
import torch
from src.pose_rgb.dataset import LineModPoseDataset
from src.pose_rgb.model import ResNetRotation, TranslationNet
import os
from utils.load_data import load_model
from src.pose_rgb.pose_utils import inverse_pinhole_projection

def visualize_RGB_random_samples(        
        checkpoint_dir: str,
        dataset_root: str,
        device: str,
        num_samples: int = 3):
    
    def prediction_function(model, device, batch):
        (model_rot, model_trans) = model

        imgs = torch.stack([sample['image'] for sample in batch], dim=0).to(device)
        bbox_info = torch.stack([sample['bbox_info'] for sample in batch], dim=0).to(device)
        bbox_center = torch.stack([sample['bbox_center'] for sample in batch], dim=0).to(device)
        cam_K = torch.stack([sample['cam_K'] for sample in batch], dim=0).to(device)
        original_width = torch.tensor([sample['original_width'] for sample in batch], dtype=torch.float32).to(device)
        original_height = torch.tensor([sample['original_height'] for sample in batch], dtype=torch.float32).to(device)

                                    
        with torch.no_grad():
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

    def gt_func(device, batch):
        gt_rotations = torch.stack([sample['rotation'] for sample in batch], dim=0).to(device)
        gt_translations = torch.stack([sample['translation'] for sample in batch], dim=0).to(device)
        return gt_rotations, gt_translations


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = LineModPoseDataset(
        root_dir=dataset_root,
        split="test",
        verbose=False
    )

    # Setup projection utils
    setup_projection_utils(dataset_root)

    model_rot_path = os.path.join(checkpoint_dir, "best_model_rot.pth")
    model_trans_path = os.path.join(checkpoint_dir, "best_model_trans.pth")
    model_rot = load_model(model_rot_path, device, ResNetRotation, freeze_backbone=False)
    model_trans = load_model(model_trans_path, device, TranslationNet)
    model = (model_rot, model_trans)

    visualize_random_samples(
        model, 
        test_dataset, 
        device, 
        inference_func=prediction_function, 
        gt_func=gt_func, 
        num_samples=num_samples,
        model_name='RGB'
    )