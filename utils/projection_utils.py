from utils.linemod_config import LineModConfig, get_linemod_config

config = None

def setup_projection_utils(dataset_root: str):
    """
    Setup projection utilities with the given dataset root.
    
    Args:
        dataset_root: Path to the LineMod dataset root
    """
    global config
    config = get_linemod_config(dataset_root)


import cv2
import numpy as np
from typing import Union
import torch
from os import path

from src.pose_rgb.pose_utils import quaternion_to_rotation_matrix

def project_points_to_2dimage(points_3d, cam_K, R_quat, t_vec):
    """
    Projects 3D points onto the 2D image plane using the camera intrinsic matrix and extrinsic parameters.

    parameters:
    - points_3d: Nx3 array of 3D points in world coordinates.
    - cam_K: 3x3 camera intrinsic matrix.
    - R_quat: (4,) Quaternion representing rotation from world to camera coordinates.
    - t_vec: (3,) translation vector from world to camera coordinates.
    """

    if hasattr(R_quat, 'numpy'):
        R_quat = R_quat.numpy()
    if hasattr(cam_K, 'numpy'):
        cam_K = cam_K.numpy()
    if hasattr(t_vec, 'numpy'):
        t_vec = t_vec.numpy()
    if hasattr(points_3d, 'numpy'):
        points_3d = points_3d.numpy()

    # Ensure points_3d is Nx3
    if points_3d.shape[1] != 3 and points_3d.shape[0] == 3:
        points_3d = points_3d.T
    if points_3d.shape[1] != 3:
        raise ValueError("points_3d must be of shape Nx3")

    R = quaternion_to_rotation_matrix(R_quat) # Convert quaternion to rotation matrix
    t_vec = t_vec.reshape((3, 1))   # Ensure t_vec is a column vector

    # Transform points from world to camera coordinates
    points_cam = R @ points_3d.T + t_vec  # 3xN
    points_cam = points_cam.T  # Nx3

    # Project points onto the image plane
    points_2d_homogeneous = cam_K @ points_cam.T  # 3xN
    points_2d_homogeneous = points_2d_homogeneous.T  # Nx3

    # Convert from homogeneous to Cartesian coordinates
    epsilon = 1e-8
    points_2d = points_2d_homogeneous[:, :2] / (points_2d_homogeneous[:, 2][:, np.newaxis] + epsilon)  # Nx2

    return points_2d



def draw_3d_bbox(image: np.ndarray, bbox_2d, color=(0, 255, 0), thickness=2):
    """
    Draws the projected 3D bounding box on the image as a wireframe cube.

    Args:
        image: RGB image (numpy array)
        bbox_2d: (8, 2) array with projected vertices
        color: line color (R, G, B)
        thickness: line thickness
    Returns:
        image with drawn bbox
    """

    img_copy = image.copy()

    # Define the 12 edges of the cube (connections between vertices)
    # Vertices: 0-3 bottom face, 4-7 top face

    edges = [
        # Inferior base (z=min)
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Superior base (z=max)
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # Draw edges
    for (i, j) in edges:
        pt1 = tuple(bbox_2d[i].astype(int))
        pt2 = tuple(bbox_2d[j].astype(int))
        cv2.line(img_copy, pt1, pt2, color, thickness)
    
    # Optionally, draw vertices
    for i in range(8):
        pt = tuple(bbox_2d[i].astype(int))
        cv2.circle(img_copy, pt, 4, color, -1)
    
    return img_copy


def draw_3d_axes(
    image: np.ndarray,
    cam_K: np.ndarray,
    rotation_quat,
    translation,
    axis_length=0.05,
    thickness=3,
    color_axes=None
):
    """
    Draws the 3D axes (X, Y, Z) of the object on the image.

    Args:
        image: RGB image (numpy array)
        cam_K: (3, 3) camera intrinsic matrix
        rotation_quat: (4,) quaternion [w, x, y, z]
        translation: (3,) translation vector [tx, ty, tz] in METERS
        axis_length: length of the axes in meters (default 5cm)
        thickness: line thickness
        color_axes: dict with keys 'x', 'y', 'z' and tuple RGB values (optional)
    Returns:
        image with drawn axes
    """
    img_copy = image.copy()
    # Origin and axes in the object frame
    origin = np.array([[0, 0, 0]])  # Origin
    axes_3d = np.array([
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ])
    points_3d = np.vstack([origin, axes_3d])  # (4, 3)
    points_2d = project_points_to_2dimage(points_3d, cam_K, rotation_quat, translation)
    origin_2d = np.nan_to_num(points_2d[0]).astype(int)
    x_axis_2d = np.nan_to_num(points_2d[1]).astype(int)
    y_axis_2d = np.nan_to_num(points_2d[2]).astype(int)
    z_axis_2d = np.nan_to_num(points_2d[3]).astype(int)

    # Use custom or default colors
    if color_axes is None:
        color_axes = {'x': (255, 0, 0), 'y': (0, 255, 0), 'z': (0, 0, 255)}

    cv2.line(img_copy, tuple(origin_2d), tuple(x_axis_2d), color_axes['x'], thickness)
    cv2.line(img_copy, tuple(origin_2d), tuple(y_axis_2d), color_axes['y'], thickness)
    cv2.line(img_copy, tuple(origin_2d), tuple(z_axis_2d), color_axes['z'], thickness)
    cv2.putText(img_copy, 'X', tuple(x_axis_2d + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_axes['x'], 2)
    cv2.putText(img_copy, 'Y', tuple(y_axis_2d + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_axes['y'], 2)
    cv2.putText(img_copy, 'Z', tuple(z_axis_2d + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_axes['z'], 2)
    return img_copy


# ============================================================================
# High-level projection functions for GT and predictions
# ============================================================================

def project_model_points(points_3d, rotation_quat, translation, cam_K):
    """
    Project 3D model points to 2D image plane.
    Use this for both GT and predictions.
    
    Args:
        points_3d: (N, 3) array of 3D points in METERS
        rotation_quat: (4,) quaternion [w, x, y, z] (GT or predicted)
        translation: (3,) translation vector in METERS (GT or predicted)
        cam_K: (3, 3) camera intrinsic matrix
    
    Returns:
        points_2d: (N, 2) projected 2D points
    
    Example:
        # Project points directly
        points_2d = project_model_points(points_3d, rot, trans, cam_K)
    """
    # Project to 2D
    points_2d = project_points_to_2dimage(points_3d, cam_K, rotation_quat, translation)
    
    return points_2d


def project_model_points_from_object(object_id: int, rotation_quat, translation, cam_K):
    """
    Load 3D model points from object ID and project to 2D image plane.
    Helper function that loads model points and calls project_model_points.
    
    Args:
        object_id: Object ID (1-15)
        rotation_quat: (4,) quaternion [w, x, y, z] (GT or predicted)
        translation: (3,) translation vector in METERS (GT or predicted)
        cam_K: (3, 3) camera intrinsic matrix
    
    Returns:
        points_2d: (N, 2) projected 2D points
    
    Example:
        # GT projection
        gt_points_2d = project_model_points_from_object(obj_id, gt_rot, gt_trans, cam_K)
        
        # Predicted projection
        pred_points_2d = project_model_points_from_object(obj_id, pred_rot, pred_trans, cam_K)
    """
    if config is None:
        raise RuntimeError("Config not initialized! Call setup_projection_utils() first")
    
    # Load 3D model points in meters
    model_3d = config.get_model_3d(object_id, unit='m')
    
    # Project to 2D using base function
    return project_model_points(model_3d, rotation_quat, translation, cam_K)


def project_bbox_corners(object_id: int, rotation_quat, translation, cam_K):
    """
    Project 3D bounding box corners to 2D image plane.
    Use this for both GT and predictions.
    
    Args:
        object_id: Object ID (1-15)
        rotation_quat: (4,) quaternion [w, x, y, z] (GT or predicted)
        translation: (3,) translation vector in METERS (GT or predicted)
        cam_K: (3, 3) camera intrinsic matrix
    
    Returns:
        bbox_2d: (8, 2) projected 2D bbox corners
    
    Example:
        # GT bbox
        gt_bbox_2d = project_bbox_corners(obj_id, gt_rot, gt_trans, cam_K)
        
        # Predicted bbox
        pred_bbox_2d = project_bbox_corners(obj_id, pred_rot, pred_trans, cam_K)
    """
    if config is None:
        raise RuntimeError("Config not initialized! Call setup_projection_utils() first")
    
    # Load 3D bbox in meters
    bbox_3d = config.get_3d_bbox(object_id, unit='m')
    
    # Project to 2D
    bbox_2d = project_points_to_2dimage(bbox_3d, cam_K, rotation_quat, translation)
    
    return bbox_2d


def project_model_center(object_id: int, rotation_quat, translation, cam_K):
    """
    Project 3D model center to 2D image plane.
    Use this for both GT and predictions.
    
    Args:
        object_id: Object ID (1-15)
        rotation_quat: (4,) quaternion [w, x, y, z] (GT or predicted)
        translation: (3,) translation vector in METERS (GT or predicted)
        cam_K: (3, 3) camera intrinsic matrix
    
    Returns:
        center_2d: (2,) projected 2D center point
    
    Example:
        # GT center
        gt_center_2d = project_model_center(obj_id, gt_rot, gt_trans, cam_K)
        
        # Predicted center
        pred_center_2d = project_model_center(obj_id, pred_rot, pred_trans, cam_K)
    """
    if config is None:
        raise RuntimeError("Config not initialized! Call setup_projection_utils() first")
    
    # Load 3D model center in meters
    center_3d = config.get_model_center(object_id, unit='m')
    
    # Project to 2D
    center_2d = project_points_to_2dimage(center_3d.reshape(1, 3), cam_K, rotation_quat, translation)[0]
    
    return center_2d


def visualize_pose_comparison(image, object_id, cam_K, 
                              gt_rotation, gt_translation,
                              pred_rotation, pred_translation):
    """
    Visualize GT and predicted poses side by side on the same image.
    
    Args:
        image: RGB image (numpy array)
        object_id: Object ID
        cam_K: Camera intrinsic matrix
        gt_rotation: GT quaternion
        gt_translation: GT translation in meters
        pred_rotation: Predicted quaternion
        pred_translation: Predicted translation in meters
    
    Returns:
        image with both GT (cyan) and predicted (red) visualizations
    """
    if config is None:
        raise RuntimeError("Config not initialized! Call setup_projection_utils() first")
    
    img_vis = image.copy()
    
    # Project GT (cyan bbox, blue axes)
    gt_bbox_2d = project_bbox_corners(object_id, gt_rotation, gt_translation, cam_K)
    img_vis = draw_3d_bbox(img_vis, gt_bbox_2d, color=(255, 255, 0), thickness=2) # Cyan in BGR
    # Axes color for GT: blue/cyan shades
    gt_axes_colors = {'x': (255, 0, 0), 'y': (255, 255, 0), 'z': (255, 128, 0)} # Blue, Cyan, Light Blue
    img_vis = draw_3d_axes(img_vis, cam_K, gt_rotation, gt_translation, axis_length=0.08, thickness=2, color_axes=gt_axes_colors)

    # Project Prediction (magenta bbox, orange axes)
    pred_bbox_2d = project_bbox_corners(object_id, pred_rotation, pred_translation, cam_K)
    img_vis = draw_3d_bbox(img_vis, pred_bbox_2d, color=(255, 0, 255), thickness=2) # Magenta in BGR
    # Axes color for prediction: orange/yellow shades
    pred_axes_colors = {'x': (0, 128, 255), 'y': (255, 0, 255), 'z': (0, 0, 255)} # Orange, Magenta, Red
    img_vis = draw_3d_axes(img_vis, cam_K, pred_rotation, pred_translation, axis_length=0.08, thickness=2, color_axes=pred_axes_colors)

    # Add legend
    cv2.putText(img_vis, 'GT: bbox cyan, axes blue/cyan', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img_vis, 'Pred: bbox magenta, axes orange/magenta', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return img_vis

def get_image(image_path: str):
    """
    Loads an image from the given path in RGB format.

    Args:
        image_path: Path to the image file.
    Returns:
        image: RGB image as a numpy array.
    """

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

import random
import matplotlib.pyplot as plt

def get_image_from_sample(sample: dict):
    """
    Loads the RGB image from a dataset sample dictionary.

    Args:
        sample: Dictionary containing sample data, including 'object_id' and 'img_id'.
    Returns:
        image: RGB image as a numpy array.
    """
    root_dir = config.dataset_root
    img_path = path.join(root_dir, "data", f"{sample['object_id']:02d}", "rgb", f"{sample['img_id']:04d}.png")
    return get_image(img_path)

def visualize_random_samples(model, dataset, device, inference_func, num_samples=5):
    samples = random.sample(range(len(dataset)), num_samples)

    batch = [dataset[i] for i in samples]
    
    pred_rotations, pred_translations = inference_func(model, device, batch)
    pred_rotations = pred_rotations.cpu().numpy()
    pred_translations = pred_translations.cpu().numpy()

    # Batch num_samples images
    for i, idx in enumerate(samples):
        img_path = batch[i]['img_path']
        image_rgb = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        img_vis = visualize_pose_comparison(
            image_rgb,
            object_id=batch[i]['object_id'],
            cam_K=batch[i]['cam_K'].numpy(),
            gt_rotation=batch[i]['rotation'].numpy(),
            gt_translation=batch[i]['translation'].numpy(),
            pred_rotation=pred_rotations[i],
            pred_translation=batch[i]['3D_center'].numpy()
        )

        img_vis_rgb = img_vis
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img_vis_rgb)
        plt.axis('off')
        plt.title(f"Sample {idx}", fontsize=10)
    plt.tight_layout()
    plt.show()