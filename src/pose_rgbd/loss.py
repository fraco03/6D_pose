import torch
import torch.nn as nn

class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, pred_q, target_q):
        """
        Args:
            pred_q (torch.Tensor): Predicted quaternions, shape (N, 4)
            target_q (torch.Tensor): Ground truth quaternions, shape (N, 4)
        
        Returns:
            torch.Tensor: Geodesic loss value
        """
        # Normalize quaternions
        pred_q = pred_q / (torch.norm(pred_q, dim=1, keepdim=True) + 1e-8)
        target_q = target_q / (torch.norm(target_q, dim=1, keepdim=True) + 1e-8)

        # Dot product between predicted and target quaternions
        dot_product = torch.sum(pred_q * target_q, dim=1)

        # Clamp dot product to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Compute geodesic loss (angle between quaternions)
        loss = torch.acos(dot_product).mean()

        return loss