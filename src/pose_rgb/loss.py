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

class TranslationLoss(nn.Module):
    """
    Loss function specialized for 3D Translation Regression.
    Handles the scale difference between Pixel offsets (XY) and Depth (Z).
    """
    def __init__(self, beta=1.0, z_weight=10.0):
        """
        Args:
            beta (float): Threshold for SmoothL1.
            z_weight (float): Multiplier for the depth loss to balance it with pixel loss.
                              Since Z is in meters (small numbers like 0.5) and XY are often 
                              in pixels or larger scales, we boost Z's importance.
        """
        super(TranslationLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=beta)
        self.z_weight = z_weight

    def forward(self, pred_trans, gt_trans):
        """
        Args:
            pred_trans: [Batch, 3] -> (dx, dy, z)
            gt_trans:   [Batch, 3] -> (dx, dy, z)
        """
        # Separate components
        pred_xy = pred_trans[:, :2] # dx, dy
        pred_z  = pred_trans[:, 2]  # z
        
        gt_xy = gt_trans[:, :2]
        gt_z  = gt_trans[:, 2]
        
        # 1. Loss XY (Pixel offsets)
        loss_xy = self.loss_fn(pred_xy, gt_xy)
        
        # 2. Loss Z (Depth)
        loss_z = self.loss_fn(pred_z, gt_z)
        
        # 3. Weighted Sum
        total_loss = loss_xy + (self.z_weight * loss_z)
        
        return total_loss
    
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
    