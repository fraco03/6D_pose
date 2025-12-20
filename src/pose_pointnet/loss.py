import torch
import torch.nn as nn

class MultiObjectPointMatchingLoss(nn.Module):
    def __init__(self, all_model_points: torch.Tensor):
        """
        Args:
            all_model_points (torch.Tensor): Shape (Num_Classes, N, 3).
                                             All sampled to the same number N of points.
        """
        super(MultiObjectPointMatchingLoss, self).__init__()
        self.register_buffer('point_bank', all_model_points)

    def quaternion_to_matrix(self, quats):
        # Security normalization to avoid division by zero
        quats = quats / (torch.norm(quats, dim=1, keepdim=True) + 1e-8)
        
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

    def forward(self, pred_q, pred_t, gt_q, gt_t, class_indices):
        """
        Args:
            pred_q (Batch, 4): Predicted Quaternions.
            pred_t (Batch, 3): Predicted Translations (centroid + residual).
            gt_q (Batch, 4): Ground Truth Quaternions.
            gt_t (Batch, 3): Ground Truth Translations.
            class_indices (Batch): Batch object class indices.
        """
        # 1. Recupera i punti corretti dal bank (Batch, N, 3)
        # Nota: questi punti sono solitamente centrati in (0,0,0) nel file CAD
        batch_points = self.point_bank[class_indices] 
        
        # 2. Quaternioni -> Matrici (Batch, 3, 3)
        R_pred = self.quaternion_to_matrix(pred_q)
        R_gt = self.quaternion_to_matrix(gt_q)
        
        # 3. Applica Rotazione e Traslazione
        # P_pred = (Points * R_pred^T) + t_pred
        # Unsqueeze(1) serve per fare broadcasting di t su tutti gli N punti
        pred_points_trans = torch.bmm(batch_points, R_pred.transpose(1, 2)) + pred_t.unsqueeze(1)
        
        # P_gt = (Points * R_gt^T) + t_gt
        gt_points_trans = torch.bmm(batch_points, R_gt.transpose(1, 2)) + gt_t.unsqueeze(1)
        
        # 4. Calcola la Loss (ADD)
        # Distanza Euclidea media per punto
        # Dim=2 è la dimensione delle coordinate (x,y,z)
        dist = torch.norm(pred_points_trans - gt_points_trans, dim=2) # (Batch, N)
        loss = torch.mean(dist) # Media su tutto il batch e tutti i punti
        
        return loss