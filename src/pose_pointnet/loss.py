import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MultiObjectPointMatchingLoss(nn.Module):
    def __init__(self, all_model_points: torch.Tensor, symmetric_class_ids: list = [10, 11]):
        """
        Args:
            all_model_points (torch.Tensor): Shape (Num_Classes, N, 3).
                                             Points sampled from CAD models.
            symmetric_class_ids (list): List of class indices that are symmetric
                                        (e.g., [8, 10] for eggbox and glue in LineMOD).
        """
        super(MultiObjectPointMatchingLoss, self).__init__()
        # Register buffer
        self.register_buffer('point_bank', all_model_points)
        self.symmetric_ids = set(symmetric_class_ids)

    def quaternion_to_matrix(self, quats):
        """
        Converts quaternions (w, x, y, z) to 3x3 rotation matrices.
        """
        # Security normalization
        quats = quats / (torch.norm(quats, dim=1, keepdim=True) + 1e-8)
        
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        x2, y2, z2 = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        R = torch.stack([
            1 - 2*(y2 + z2),     2*(xy - wz),     2*(xz + wy),
                2*(xy + wz), 1 - 2*(x2 + z2),     2*(yz - wx),
                2*(xz - wy),     2*(yz + wx), 1 - 2*(x2 + y2)
        ], dim=1).reshape(-1, 3, 3)
        return R

    def forward(self, pred_q, pred_t, gt_q, gt_t, class_indices):
        """
        Computes the ADD Loss (asymmetric) or ADD-S (symmetric).
        
        Args:
            pred_q (Batch, 4): Predicted Quaternions (w,x,y,z).
            pred_t (Batch, 3): Predicted Translations.
            gt_q (Batch, 4): Ground Truth Quaternions.
            gt_t (Batch, 3): Ground Truth Translations.
            class_indices (Batch): Class indices of objects in the batch.
        """
        # 1. Retrieve correct points from the bank
        batch_points = self.point_bank[class_indices] # (Batch, N, 3)
        
        # 2. Convert Quaternions to Matrices
        R_pred = self.quaternion_to_matrix(pred_q) # (Batch, 3, 3)
        R_gt = self.quaternion_to_matrix(gt_q)     # (Batch, 3, 3)
        
        # 3. Point Transformation (Rotation + Translation)
        # BMM expects (Batch, N, 3) x (Batch, 3, 3)^T -> (Batch, N, 3)
        pred_points_trans = torch.bmm(batch_points, R_pred.transpose(1, 2)) + pred_t.unsqueeze(1)
        gt_points_trans = torch.bmm(batch_points, R_gt.transpose(1, 2)) + gt_t.unsqueeze(1)
        
        # 4. Hybrid Loss Computation (vectorized for speed)
        # Initialize the loss vector for each batch element
        losses = torch.zeros(pred_q.shape[0], device=pred_q.device)
        
        # Identify which elements in the batch are symmetric and which are not
        # Create a boolean mask
        is_symmetric = torch.tensor([c.item() in self.symmetric_ids for c in class_indices], 
                                    device=pred_q.device, dtype=torch.bool)
        
        # --- COMPUTATION FOR ASYMMETRIC OBJECTS (ADD) ---
        if (~is_symmetric).any():
            # Point-to-point distance (exact correspondence)
            # Norm on dim=2 (xyz), then mean on dim=1 (points)
            diff = pred_points_trans[~is_symmetric] - gt_points_trans[~is_symmetric]
            add_loss = torch.mean(torch.norm(diff, dim=2), dim=1)
            losses[~is_symmetric] = add_loss

        # --- COMPUTATION FOR SYMMETRIC OBJECTS (ADD-S) ---
        if is_symmetric.any():
            # Nearest-point distance
            p_pred_sym = pred_points_trans[is_symmetric] # (M, N, 3)
            p_gt_sym = gt_points_trans[is_symmetric]     # (M, N, 3)
            
            # Compute the distance matrix between all points (pairwise distance)
            # cdist computes the distance between each vector in P1 and each vector in P2
            # Output shape: (M, N, N)
            dist_matrix = torch.cdist(p_pred_sym, p_gt_sym, p=2)
            
            # For each predicted point, find the minimum distance to GT points
            # min on dim=2 (columns, i.e., the GT points)
            min_dists, _ = torch.min(dist_matrix, dim=2) # (M, N)
            
            adds_loss = torch.mean(min_dists, dim=1) # Average over points (M)
            losses[is_symmetric] = adds_loss

        # 5. Final average over the entire batch
        return losses.mean()