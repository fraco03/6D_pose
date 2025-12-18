import torch
import torch.nn as nn

from src.pose_rgb.pose_utils import quaternion_to_rotation_matrix

class RotationLoss(nn.Module):
    def __init__(self):
        super(RotationLoss, self).__init__()

    def forward(self, pred_q, gt_q):
        # Normalizzazione di sicurezza (fondamentale)
        pred_q = torch.nn.functional.normalize(pred_q, p=2, dim=1)
        gt_q = torch.nn.functional.normalize(gt_q, p=2, dim=1)
        
        dot_product = torch.sum(pred_q * gt_q, dim=1)
        # Clamp per evitare errori numerici che portano a loss negativa
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        loss = 1.0 - torch.abs(dot_product)
        return loss.mean()
    
class PoseMatchingLoss(nn.Module):
    """
    ROTATION ONLY pose matching loss
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred_q, gt_q, model_points):
        """
        Args:
            pred_q: (B, 4) predicted quaternions
            gt_q: (B, 4) ground truth quaternions
            model_points: (B, N, 3) object model points in object frame
        Returns:
            loss: scalar pose matching loss
        """
        gt_rot_points = self._rotate_with_quaternion(model_points, gt_q)  # (B, N, 3)
        pred_rot_points = self._rotate_with_quaternion(model_points, pred_q)  # (B, N, 3)

        diff = gt_rot_points - pred_rot_points  # (B, N, 3)
        loss = torch.mean(torch.norm(diff, dim=2))  # scalar
        return loss

    
    def _rotate_with_quaternion(self, points, quaternion):
        """
        Rotate points with quaternion
        Args:
            points: (B, N, 3)
            quaternion: (B, 4)
        Returns:
            rotated_points: (B, N, 3)
        """
        batch_size, num_points, _ = points.shape

        w = quaternion[:, 0].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        q_vec = quaternion[:, 1:].unsqueeze(1)        # (B, 1, 3)

        # Cross product rquires points to be (B, 3, N)
        q_vec = q_vec.expand(-1, num_points, -1)  # (B, N, 3)

        # Optimized quaternion rotation
        t = 2.0 * torch.cross(q_vec, points, dim=-1)          # (B, N, 3)
        res = points + w * t + torch.cross(q_vec, t, dim=-1)  # (B, N, 3)

        return res
    
    