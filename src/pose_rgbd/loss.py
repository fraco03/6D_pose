import torch
import torch.nn as nn

class MultiObjectPointMatchingLoss(nn.Module):
    def __init__(self, all_model_points: torch.Tensor):
        """
        Args:
            all_model_points (torch.Tensor): Shape (Num_Classes, N, 3).
                                             Contains point clouds for all objects.
        """
        super(MultiObjectPointMatchingLoss, self).__init__()
        # Register the bank of points
        self.register_buffer('point_bank', all_model_points)

    def quaternion_to_matrix(self, quats):
        # (Same standard conversion function as before)
        x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        x2, y2, z2 = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        R = torch.stack([
            1 - 2*(y2 + z2),     2*(xy - wz),     2*(xz + wy),
                2*(xy + wz), 1 - 2*(x2 + z2),     2*(yz - wx),
                2*(xz - wy),     2*(yz + wx), 1 - 2*(x2 + y2)
        ], dim=1).reshape(-1, 3, 3)
        return R

    def forward(self, pred_q, gt_q, class_indices):
        """
        Args:
            pred_q (Batch, 4): Predicted Quaternions.
            gt_q (Batch, 4): Ground Truth Quaternions.
            class_indices (Batch): Tensor of integers (0 to Num_Classes-1).
                                   Indicates which object is in the image.
        """
        # 1. Select the correct 3D points for each item in the batch
        # This is the magic part:
        # self.point_bank[class_indices] automatically creates a batch of shape (Batch, N, 3)
        # where the first element gets points for its class, second for its class, etc.
        batch_points = self.point_bank[class_indices] 
        
        # 2. Convert Quaternions to Matrices
        R_pred = self.quaternion_to_matrix(pred_q)
        R_gt = self.quaternion_to_matrix(gt_q)
        
        # 3. Apply Rotation (Batch Matrix Multiplication)
        # batch_points shape: (Batch, N, 3)
        # R shape: (Batch, 3, 3)
        pred_points_rot = torch.bmm(batch_points, R_pred.transpose(1, 2))
        gt_points_rot = torch.bmm(batch_points, R_gt.transpose(1, 2))
        
        # 4. Calculate Loss
        loss = torch.mean(torch.norm(pred_points_rot - gt_points_rot, dim=2))
        
        return loss

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