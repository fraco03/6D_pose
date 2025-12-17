import numpy as np
from src.pose_rgb.pose_utils import quaternion_to_rotation_matrix

def compute_ADD_metric(model_points, gt_rotation, gt_translation,
                       pred_rotation, pred_translation):
    """
    Compute the ADD metric for a single object instance.
    Args:
        model_points: (N, 3) array of 3D model points
        gt_rotation: (3, 3) ground truth rotation matrix
        gt_translation: (3,) ground truth translation vector
        pred_rotation: (3, 3) predicted rotation matrix
        pred_translation: (3,) predicted translation vector
    Returns:
        add: Average Distance of Model Points (ADD) metric
    """

    # Transform model points with ground
    gt_points = (gt_rotation @ model_points.T).T + gt_translation  # (N, 3)
    # Transform model points with prediction
    pred_points = (pred_rotation @ model_points.T).T + pred_translation  # (N, 3)

    # Compute average distance
    distances = np.linalg.norm(gt_points - pred_points, axis=1)  # (N,)
    add = distances.mean()
    return add
    

def compute_ADD_metric_quaternion(model_points, gt_quat, gt_translation,
                                  pred_quat, pred_translation):
    """
    Compute the ADD metric for a single object instance using quaternions.
    Args:
        model_points: (N, 3) array of 3D model points
        gt_quat: (4,) ground truth quaternion (w, x, y, z)
        gt_translation: (3,) ground truth translation vector
        pred_quat: (4,) predicted quaternion (w, x, y, z)
        pred_translation: (3,) predicted translation vector
    Returns:
        add: Average Distance of Model Points (ADD) metric
    """
    gt_rotation = quaternion_to_rotation_matrix(gt_quat)
    pred_rotation = quaternion_to_rotation_matrix(pred_quat)
    return compute_ADD_metric(model_points, gt_rotation, gt_translation,
                              pred_rotation, pred_translation)


def compute_ADD_S_metric(model_points, gt_rotation, gt_translation,
                         pred_rotation, pred_translation):
    """
    Compute ADD-S (Symmetric) metric for symmetric objects.
    For each GT point, finds the closest predicted point.
    
    Args:
        model_points: (N, 3) array of 3D model points in meters
        gt_rotation: (3, 3) ground truth rotation matrix
        gt_translation: (3,) ground truth translation vector in meters
        pred_rotation: (3, 3) predicted rotation matrix
        pred_translation: (3,) predicted translation vector in meters
    
    Returns:
        add_s: Average closest point distance (ADD-S) metric in meters
    """
    # Transform model points with GT and prediction
    gt_points = (gt_rotation @ model_points.T).T + gt_translation
    pred_points = (pred_rotation @ model_points.T).T + pred_translation
    
    # For each GT point, find the closest predicted point
    from scipy.spatial.distance import cdist
    distances = cdist(gt_points, pred_points, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    add_s = min_distances.mean()
    return add_s


def compute_ADD_S_metric_quaternion(model_points, gt_quat, gt_translation,
                                    pred_quat, pred_translation):
    """
    Compute ADD-S metric using quaternions (for symmetric objects).
    
    Args:
        model_points: (N, 3) array of 3D model points
        gt_quat: (4,) ground truth quaternion (w, x, y, z)
        gt_translation: (3,) ground truth translation vector
        pred_quat: (4,) predicted quaternion (w, x, y, z)
        pred_translation: (3,) predicted translation vector
    
    Returns:
        add_s: ADD-S metric
    """
    gt_rotation = quaternion_to_rotation_matrix(gt_quat)
    pred_rotation = quaternion_to_rotation_matrix(pred_quat)
    return compute_ADD_S_metric(model_points, gt_rotation, gt_translation,
                                pred_rotation, pred_translation)
