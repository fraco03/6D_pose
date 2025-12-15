import numpy as np
import torch

def convert_rotation_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion representation"""

    assert rotation_matrix.shape == (3, 3), "Input must be a 3x3 rotation matrix."
    
    m = rotation_matrix
    trace = np.trace(m)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - trace)
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - trace)
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - trace)
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    
    quaternion = np.array([w, x, y, z])
    quaternion /= (np.linalg.norm(quaternion) + 1e-8)  # Normalize the quaternion
    return quaternion

def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix representation"""

    assert quaternion.shape == (4,), "Input must be a quaternion of shape (4,)."
    
    w, x, y, z = quaternion
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])
    return R
    
def inverse_pinhole_projection(
    crop_center: torch.Tensor, 
    deltas: torch.Tensor, 
    z: torch.Tensor, 
    cam_K: torch.Tensor
) -> torch.Tensor:
    """
    Converts 2D predictions (pixel offsets) into Real 3D Coordinates.
    
    It implements the Inverse Pinhole Camera Model formula:
    X = (u - cx_optical) * Z / fx
    Y = (v - cy_optical) * Z / fy
    
    The unit of measurement for X and Y will be identical to the unit of Z 
    (e.g., if Z is in meters, X and Y will be in meters).

    Args:
        crop_center (torch.Tensor): Absolute center of the crop in pixels [cx_crop, cy_crop].
                                    Shape: (Batch_Size, 2)
        deltas (torch.Tensor): Predicted offsets from the crop center [delta_x, delta_y].
                               Shape: (Batch_Size, 2)
        z (torch.Tensor): Predicted Depth (Z).
                          Shape: (Batch_Size, 1) or (Batch_Size,)
        cam_K (torch.Tensor): Camera Intrinsic Matrix.
                              Shape: (Batch_Size, 3, 3)
                              Structure: [[fx,  0, cx],
                                          [ 0, fy, cy],
                                          [ 0,  0,  1]]

    Returns:
        torch.Tensor: Absolute 3D Coordinates [X, Y, Z].
                      Shape: (Batch_Size, 3)
    """
    
    # 1. Calculate the final absolute pixel coordinates (u, v) on the image plane.
    #    u = crop_center_x + predicted_offset_x
    #    v = crop_center_y + predicted_offset_y
    u = crop_center[:, 0] + deltas[:, 0]
    v = crop_center[:, 1] + deltas[:, 1]
    
    # 2. Extract Intrinsic Parameters from the Camera Matrix K
    fx = cam_K[:, 0, 0]      # Focal length X
    fy = cam_K[:, 1, 1]      # Focal length Y
    cx_optical = cam_K[:, 0, 2] # Optical Center X (Principal Point)
    cy_optical = cam_K[:, 1, 2] # Optical Center Y (Principal Point)
    
    # 3. Ensure 'z' has the correct shape for broadcasting
    #    If z is (Batch,), we reshape it to (Batch, 1) to multiply correctly with u and v.
    if z.dim() == 1:
        z = z.view(-1)
        
    # 4. Apply Inverse Pinhole Projection
    #    Geometric Formula: X = (Pixel_Coord - Optical_Center) * Depth / Focal_Length
    X = (u - cx_optical) * z / fx
    Y = (v - cy_optical) * z / fy
    
    # 5. Stack the components to form the final 3D vector [X, Y, Z]
    #    dim=1 ensures we stack them side-by-side: (Batch, 3)
    real_3d_coords = torch.stack([X, Y, z], dim=1)
    
    return real_3d_coords