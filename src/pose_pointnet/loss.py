import torch
import torch.nn as nn

# class MultiObjectPointMatchingLoss(nn.Module):
#     def __init__(self, all_model_points: torch.Tensor):
#         """
#         Args:
#             all_model_points (torch.Tensor): Shape (Num_Classes, N, 3).
#                                              All sampled to the same number N of points.
#         """
#         super(MultiObjectPointMatchingLoss, self).__init__()
#         self.register_buffer('point_bank', all_model_points)

#     def quaternion_to_matrix(self, quats):
#         # Security normalization to avoid division by zero
#         quats = quats / (torch.norm(quats, dim=1, keepdim=True) + 1e-8)
        
#         x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
#         x2, y2, z2 = x*x, y*y, z*z
#         xy, xz, yz = x*y, x*z, y*z
#         wx, wy, wz = w*x, w*y, w*z
        
#         R = torch.stack([
#             1 - 2*(y2 + z2),     2*(xy - wz),     2*(xz + wy),
#                 2*(xy + wz), 1 - 2*(x2 + z2),     2*(yz - wx),
#                 2*(xz - wy),     2*(yz + wx), 1 - 2*(x2 + y2)
#         ], dim=1).reshape(-1, 3, 3)
#         return R

#     def forward(self, pred_q, pred_t, gt_q, gt_t, class_indices):
#         """
#         Args:
#             pred_q (Batch, 4): Predicted Quaternions.
#             pred_t (Batch, 3): Predicted Translations (centroid + residual).
#             gt_q (Batch, 4): Ground Truth Quaternions.
#             gt_t (Batch, 3): Ground Truth Translations.
#             class_indices (Batch): Batch object class indices.
#         """
#         # 1. Recupera i punti corretti dal bank (Batch, N, 3)
#         # Nota: questi punti sono solitamente centrati in (0,0,0) nel file CAD
#         batch_points = self.point_bank[class_indices] 
        
#         # 2. Quaternioni -> Matrici (Batch, 3, 3)
#         R_pred = self.quaternion_to_matrix(pred_q)
#         R_gt = self.quaternion_to_matrix(gt_q)
        
#         # 3. Applica Rotazione e Traslazione
#         # P_pred = (Points * R_pred^T) + t_pred
#         # Unsqueeze(1) serve per fare broadcasting di t su tutti gli N punti
#         pred_points_trans = torch.bmm(batch_points, R_pred.transpose(1, 2)) + pred_t.unsqueeze(1)
        
#         # P_gt = (Points * R_gt^T) + t_gt
#         gt_points_trans = torch.bmm(batch_points, R_gt.transpose(1, 2)) + gt_t.unsqueeze(1)
        
#         # 4. Calcola la Loss (ADD)
#         # Distanza Euclidea media per punto
#         # Dim=2 è la dimensione delle coordinate (x,y,z)
#         dist = torch.norm(pred_points_trans - gt_points_trans, dim=2) # (Batch, N)
#         loss = torch.mean(dist) # Media su tutto il batch e tutti i punti
        
#         return loss

import torch
import torch.nn as nn

class MultiObjectPointMatchingLoss(nn.Module):
    def __init__(self, all_model_points: torch.Tensor, symmetric_class_ids: list = [10, 11]):
        """
        Args:
            all_model_points (torch.Tensor): Shape (Num_Classes, N, 3).
                                             Punti campionati dai modelli CAD.
            symmetric_class_ids (list): Lista degli indici di classe che sono simmetrici
                                        (es. [8, 10] per eggbox e glue in LineMOD).
        """
        super(MultiObjectPointMatchingLoss, self).__init__()
        # Register buffer
        self.register_buffer('point_bank', all_model_points)
        self.symmetric_ids = set(symmetric_class_ids)

    def quaternion_to_matrix(self, quats):
        """
        Converte quaternioni (w, x, y, z) in matrici di rotazione 3x3.
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
        Calcola la Loss ADD (asimmetrici) o ADD-S (simmetrici).
        
        Args:
            pred_q (Batch, 4): Predicted Quaternions (w,x,y,z).
            pred_t (Batch, 3): Predicted Translations.
            gt_q (Batch, 4): Ground Truth Quaternions.
            gt_t (Batch, 3): Ground Truth Translations.
            class_indices (Batch): Indici delle classi degli oggetti nel batch.
        """
        # 1. Recupera i punti corretti dal bank
        batch_points = self.point_bank[class_indices] # (Batch, N, 3)
        
        # 2. Converti Quaternioni in Matrici
        R_pred = self.quaternion_to_matrix(pred_q) # (Batch, 3, 3)
        R_gt = self.quaternion_to_matrix(gt_q)     # (Batch, 3, 3)
        
        # 3. Trasformazione Punti (Rotazione + Traslazione)
        # BMM vuole (Batch, N, 3) x (Batch, 3, 3)^T -> (Batch, N, 3)
        pred_points_trans = torch.bmm(batch_points, R_pred.transpose(1, 2)) + pred_t.unsqueeze(1)
        gt_points_trans = torch.bmm(batch_points, R_gt.transpose(1, 2)) + gt_t.unsqueeze(1)
        
        # 4. Calcolo Loss Ibrida (vettorializzato per velocità)
        # Inizializziamo il vettore delle loss per elemento del batch
        losses = torch.zeros(pred_q.shape[0], device=pred_q.device)
        
        # Identifichiamo quali elementi nel batch sono simmetrici e quali no
        # Creiamo una maschera booleana
        is_symmetric = torch.tensor([c.item() in self.symmetric_ids for c in class_indices], 
                                    device=pred_q.device, dtype=torch.bool)
        
        # --- CALCOLO PER ASIMMETRICI (ADD) ---
        if (~is_symmetric).any():
            # Distanza punto-a-punto (corrispondenza esatta)
            # Norm su dim=2 (xyz), poi mean su dim=1 (punti)
            diff = pred_points_trans[~is_symmetric] - gt_points_trans[~is_symmetric]
            add_loss = torch.mean(torch.norm(diff, dim=2), dim=1)
            losses[~is_symmetric] = add_loss

        # --- CALCOLO PER SIMMETRICI (ADD-S) ---
        if is_symmetric.any():
            # Distanza punto-più-vicino
            p_pred_sym = pred_points_trans[is_symmetric] # (M, N, 3)
            p_gt_sym = gt_points_trans[is_symmetric]     # (M, N, 3)
            
            # Calcoliamo la matrice delle distanze tra tutti i punti (pairwise distance)
            # cdist calcola la distanza tra ogni vettore di P1 e ogni vettore di P2
            # Output shape: (M, N, N)
            dist_matrix = torch.cdist(p_pred_sym, p_gt_sym, p=2)
            
            # Per ogni punto predetto, troviamo la distanza minima tra i punti GT
            # min su dim=2 (colonne, cioè i punti GT)
            min_dists, _ = torch.min(dist_matrix, dim=2) # (M, N)
            
            adds_loss = torch.mean(min_dists, dim=1) # Media sui punti (M)
            losses[is_symmetric] = adds_loss

        # 5. Media finale su tutto il batch
        return losses.mean()