import torch
import torch.nn as nn

class RotationLoss(nn.Module):
    """
    Loss function specialized for Quaternion Regression.
    
    It calculates the Geodesic distance (angular difference) between 
    the predicted quaternion and the ground truth quaternion.
    
    Mathematical formulation:
    L = 1 - |<q_pred, q_gt>|
    
    We take the absolute value of the dot product to handle the 
    'double cover' property of quaternions (q and -q represent the same rotation).
    """
    def __init__(self):
        super(RotationLoss, self).__init__()

    def forward(self, pred_q, gt_q):
        """
        Args:
            pred_q (torch.Tensor): Predicted quaternions. Shape (Batch, 4).
                                   Expects normalized quaternions (unit length).
            gt_q (torch.Tensor): Ground Truth quaternions. Shape (Batch, 4).
        
        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # 1. Calculate Dot Product
        # q1 . q2 = w1*w2 + x1*x2 + y1*y2 + z1*z2
        dot_product = torch.sum(pred_q * gt_q, dim=1)
        
        # 2. Calculate angular error
        # We assume quaternions are unit length.
        # Ideally, dot product is 1.0 (or -1.0) if rotations are identical.
        # Loss becomes 0 when rotations are identical.
        loss = 1.0 - torch.abs(dot_product)
        
        # 3. Average over the batch
        return loss.mean()
    
class TranslationLoss(nn.Module):
    """
    Loss function specialized for 3D Translation Regression.
    
    It predicts a vector of 3 values: [delta_x, delta_y, z].
    Uses Smooth L1 Loss (Huber Loss) to be robust against outliers 
    in pixel offsets or depth estimation.
    """
    def __init__(self, beta=1.0):
        """
        Args:
            beta (float): Threshold for SmoothL1. 
                          If error < beta, it uses squared loss (L2).
                          If error >= beta, it uses linear loss (L1).
        """
        super(TranslationLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=beta)

    def forward(self, pred_trans, gt_trans):
        """
        Args:
            pred_trans (torch.Tensor): Predicted vector [dx, dy, z]. Shape (Batch, 3).
            gt_trans (torch.Tensor): Target vector [dx, dy, z]. Shape (Batch, 3).
                                     (Must be calculated from GT absolute translation).
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Simply compute the regression error
        loss = self.loss_fn(pred_trans, gt_trans)
        
        return loss
    
class CombinedPoseLoss(nn.Module):
    """
    Unified Loss function that combines Rotation and Translation errors.
    
    It computes the weighted sum:
    Total_Loss = (w_rot * Rot_Loss) + (w_trans * Trans_Loss)
    """
    def __init__(self, w_rot=1.0, w_trans=1.0):
        """
        Args:
            w_rot (float): Weight coefficient for rotation loss.
            w_trans (float): Weight coefficient for translation loss.
        """
        super(CombinedPoseLoss, self).__init__()
        self.w_rot = w_rot
        self.w_trans = w_trans
        
        # Instantiate the specific sub-losses
        self.rot_criterion = RotationLoss()
        self.trans_criterion = TranslationLoss()

    def forward(self, pred_rot, gt_rot, pred_trans, gt_trans):
        """
        Calculates the combined loss.
        
        Args:
            pred_rot, gt_rot: Quaternions for rotation.
            pred_trans, gt_trans: Vectors [dx, dy, z] for translation.
            
        Returns:
            total_loss: The weighted sum (used for backprop).
            l_rot: The raw rotation loss (for logging/monitoring).
            l_trans: The raw translation loss (for logging/monitoring).
        """
        # 1. Calculate individual losses
        l_rot = self.rot_criterion(pred_rot, gt_rot)
        l_trans = self.trans_criterion(pred_trans, gt_trans)
        
        # 2. Weighted Sum
        total_loss = (self.w_rot * l_rot) + (self.w_trans * l_trans)
        
        # We return all three so you can print them separately in the progress bar
        return total_loss, l_rot, l_trans