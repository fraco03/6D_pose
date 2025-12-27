from utils.projection_utils import setup_projection_utils, visualize_random_samples
import torch
from src.pose_rgbd.dataset import LineModPoseDepthDataset
from src.pose_rgbd.model import RGBDRotationModel
import os
from utils.load_data import load_model

def visualize_RGBD_random_samples(
        checkpoint_dir: str,
        dataset_root: str,
        device: str,
        num_samples: int = 3,
        sample_indices: list = None
):
    def rgbd_inference(model, device, batch: list):
        rgb = torch.stack([sample['image'] for sample in batch]).to(device)
        depth = torch.stack([sample['depth'] for sample in batch]).to(device)

        model.eval()
        with torch.no_grad():
            pred_rot = model(rgb, depth)
        pred_trans = torch.stack([sample['3D_center'] for sample in batch]).to(device)
        return pred_rot, pred_trans
    
    def gt_func(device, batch: list):
        gt_rot = torch.stack([sample['rotation'] for sample in batch]).to(device)
        gt_trans = torch.stack([sample['translation'] for sample in batch]).to(device)

        return gt_rot, gt_trans


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = LineModPoseDepthDataset(
        root_dir=dataset_root,
        split="test",
        verbose=False
    )

    # Setup projection utils
    setup_projection_utils(dataset_root)

    # Load best model
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    model = load_model(best_checkpoint_path, device, RGBDRotationModel, pretrained=False)

    visualize_random_samples(
        model, 
        test_dataset, 
        device, 
        inference_func=rgbd_inference, 
        gt_func=gt_func,
        num_samples=num_samples,
        model_name='RGB-D',
        sample_indices=sample_indices
    )