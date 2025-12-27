import torch
from utils.projection_utils import setup_projection_utils, visualize_random_samples
from src.dense_fusion.dataset import DenseFusionLineModDataset
from src.dense_fusion.model import FusionPoseModel 
from utils.load_data import load_model

def visualize_densefusion_random_samples(
        checkpoint_dir: str,
        dataset_root: str,
        device: str,
        num_samples: int = 3,
        sample_indices: list = None
):
    # Inference function for DenseFusion model
    def fusion_inference(model, device, batch: list):
        # 1. Stack
        points = torch.stack([sample['points'] for sample in batch]).to(device)
        images = torch.stack([sample['rgb'] for sample in batch]).to(device) 
        centroids = torch.stack([sample['centroid'] for sample in batch]).to(device)

        model.eval()
        with torch.no_grad():
            # 2. Forward pass
            pred_rot, pred_t_res = model(points, images)

        # 3. Absolute translation
        pred_trans = centroids + pred_t_res

        return pred_rot, pred_trans
    
    def gt_func(device, batch: list):
        gt_rot = torch.stack([sample['rotation'] for sample in batch]).to(device)
        gt_trans = torch.stack([sample['gt_translation'] for sample in batch]).to(device)
        return gt_rot, gt_trans
    

    # 1. Insantiate dataset
    test_dataset = DenseFusionLineModDataset(
        root_dir=dataset_root,
        split="test",
        verbose=False,
    )

    setup_projection_utils(dataset_root)

    model_path = f"{checkpoint_dir}/best_model.pth"
    
    # 2. Load model
    print(f"Loading FusionPoseModel from {model_path}...")
    model = load_model(model_path, device, FusionPoseModel)

    # 3. Visualize random samples
    visualize_random_samples(
        model,
        test_dataset,
        device,
        inference_func=fusion_inference,
        gt_func=gt_func,
        num_samples=num_samples,
        model_name='DenseFusion',
        sample_indices=sample_indices
    )