import torch
from utils.projection_utils import setup_projection_utils, visualize_random_samples
from src.dense_fusion.dataset import DenseFusionLineModDataset
from src.dense_fusion.model import FusionPoseModel 
from utils.load_data import load_model

def visualize_densefusion_random_samples(
        checkpoint_dir: str,
        dataset_root: str,
        device: str,
        num_samples: int = 3
):
    # Funzione di inferenza aggiornata per DenseFusion
    def fusion_inference(model, device, batch: list):
        # 1. Stack dei dati
        points = torch.stack([sample['points'] for sample in batch]).to(device)
        images = torch.stack([sample['rgb'] for sample in batch]).to(device) # IMPORTANTE: Carica RGB
        centroids = torch.stack([sample['centroid'] for sample in batch]).to(device)

        model.eval()
        with torch.no_grad():
            # 2. Forward pass con ENTRAMBI gli input
            pred_rot, pred_t_res = model(points, images)

        # 3. Ricostruzione traslazione assoluta
        pred_trans = centroids + pred_t_res

        return pred_rot, pred_trans
    
    def gt_func(device, batch: list):
        gt_rot = torch.stack([sample['rotation'] for sample in batch]).to(device)
        gt_trans = torch.stack([sample['gt_translation'] for sample in batch]).to(device)
        return gt_rot, gt_trans
    

    # 1. Usa il Dataset corretto (DenseFusion) che restituisce 'rgb'
    test_dataset = DenseFusionLineModDataset(
        root_dir=dataset_root,
        split="test",
        verbose=False,
        # Assicurati che il resize corrisponda a quello usato in training se necessario, 
        # ma visualize_random_samples di solito gestisce batch=1 o liste.
    )

    setup_projection_utils(dataset_root)

    model_path = f"{checkpoint_dir}/best_model.pth"
    
    # 2. Carica la classe modello corretta (FusionPoseModel)
    print(f"Loading FusionPoseModel from {model_path}...")
    model = load_model(model_path, device, FusionPoseModel)

    # 3. Lancia la visualizzazione
    visualize_random_samples(
        model,
        test_dataset,
        device,
        inference_func=fusion_inference, # Usa la nuova funzione
        gt_func=gt_func,
        num_samples=num_samples,
        model_name='DenseFusion'
    )