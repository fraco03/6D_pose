import torch
from utils.projection_utils import setup_projection_utils, visualize_random_samples
from src.pose_pointnet.dataset import PointNetLineModDataset
from src.pose_pointnet.model import PointNetPoseModel
from utils.load_data import load_model

def visualize_pointnet_random_samples(
        checkpoint_dir: str,
        dataset_root: str,
        device: str,
        num_samples: int = 3
):
    def pointnet_inference(model, device, batch: list):
        points = torch.stack([sample['points'] for sample in batch]).to(device)
        centroids = torch.stack([sample['centroid'] for sample in batch]).to(device)


        model.eval()
        with torch.no_grad():
            pred_rot, pred_t_res = model(points)

        pred_trans = centroids + pred_t_res

        return pred_rot, pred_trans
    
    def gt_func(device, batch: list):
        gt_rot = torch.stack([sample['rotation'] for sample in batch]).to(device)
        gt_trans = torch.stack([sample['gt_translation'] for sample in batch]).to(device)
        
        return gt_rot, gt_trans
    

    test_dataset = PointNetLineModDataset(
        root_dir=dataset_root,
        split="test",
        verbose=False
    )

    # Setup projection utils
    setup_projection_utils(dataset_root)

    model_path = f"{checkpoint_dir}/best_model.pth"
    model = load_model(model_path, device, PointNetPoseModel)

    visualize_random_samples(
        model,
        test_dataset,
        device,
        inference_func=pointnet_inference,
        gt_func=gt_func,
        num_samples=num_samples,
        model_name='PointNet'
    )


        