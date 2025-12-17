import torch
import torch.nn as nn

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

class DisentangledTranslationLoss(nn.Module):
    """
    Separa XY (pixel/schermo) da Z (profondità metrica).
    """
    def __init__(self, use_log_z=True):
        super(DisentangledTranslationLoss, self).__init__()
        self.use_log_z = use_log_z
        self.l1 = nn.SmoothL1Loss(reduction='mean')

    def forward(self, pred_trans, gt_trans):
        # pred_trans: [dx, dy, z]
        
        # 1. Loss sui pixel (dx, dy)
        loss_xy = self.l1(pred_trans[:, :2], gt_trans[:, :2])
        
        # 2. Loss sulla profondità (Z)
        pred_z = pred_trans[:, 2]
        gt_z = gt_trans[:, 2]
        
        if self.use_log_z:
            # Log loss è meglio per la profondità perché penalizza
            # gli errori relativi (sbagliare di 10cm su 1m è grave, su 10m no)
            # Aggiungiamo epsilon per sicurezza
            loss_z = self.l1(torch.log(torch.abs(pred_z) + 1e-6), 
                             torch.log(torch.abs(gt_z) + 1e-6))
        else:
            loss_z = self.l1(pred_z, gt_z)
            
        return loss_xy, loss_z

class AutomaticWeightedLoss(nn.Module):
    """
    Bilanciamento automatico delle Loss (Kendall et al. CVPR 2018).
    Invece di cercare w_rot e w_trans a mano, la rete impara 'sx', 'sy', 'sz'.
    
    Loss = (Loss_A / (2 * sigma_A^2)) + log(sigma_A)
    
    Args:
        use_disentangled: Se True, usa DisentangledTranslationLoss (XY separato da Z).
                          Se False, usa una loss unificata per X,Y,Z (per PointNet).
    """
    def __init__(self, use_disentangled=True):
        super(AutomaticWeightedLoss, self).__init__()
        
        self.use_disentangled = use_disentangled
        
        # Parametri apprendibili (inizializzati a 0 -> sigma=1)
        # sx: varianza per la rotazione
        # st: varianza per translation (unificata) - usato solo se use_disentangled=False
        # sy: varianza per offset XY - usato solo se use_disentangled=True
        # sz: varianza per profondità Z - usato solo se use_disentangled=True
        self.sx = nn.Parameter(torch.tensor(0.0))
        
        if use_disentangled:
            self.sy = nn.Parameter(torch.tensor(0.0))
            self.sz = nn.Parameter(torch.tensor(0.0))
            self.trans_loss_fn = DisentangledTranslationLoss(use_log_z=True)
        else:
            self.st = nn.Parameter(torch.tensor(0.0))
            self.trans_loss_fn = nn.SmoothL1Loss(reduction='mean')
        
        self.rot_loss_fn = RotationLoss()

    def forward(self, pred_rot, gt_rot, pred_trans, gt_trans):
        # 1. Calcola le loss grezze
        l_rot = self.rot_loss_fn(pred_rot, gt_rot)
        
        if self.use_disentangled:
            # Modalità per pose_rgb: separa XY da Z
            l_xy, l_z = self.trans_loss_fn(pred_trans, gt_trans)
            
            # Pesatura Automatica (Multi-Task Loss)
            loss_rot_weighted = l_rot * torch.exp(-self.sx) + self.sx
            loss_xy_weighted = l_xy * torch.exp(-self.sy) + self.sy
            loss_z_weighted = l_z * torch.exp(-self.sz) + self.sz
            
            total_loss = loss_rot_weighted + loss_xy_weighted + loss_z_weighted
            
            return total_loss, l_rot, l_xy, l_z
        else:
            # Modalità per PointNet: translation unificata
            l_trans = self.trans_loss_fn(pred_trans, gt_trans)
            
            # Pesatura Automatica
            loss_rot_weighted = l_rot * torch.exp(-self.sx) + self.sx
            loss_trans_weighted = l_trans * torch.exp(-self.st) + self.st
            
            total_loss = loss_rot_weighted + loss_trans_weighted
            
            # Per compatibilità col codice esistente, ritorniamo l_trans sia per XY che Z
            return total_loss, l_rot, l_trans, l_trans