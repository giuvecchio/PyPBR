"""
pypbr.utils

This module provides utility functions for color space conversions between sRGB and linear space.
These functions are used in Physically Based Rendering (PBR) pipelines to ensure correct color 
representation during material rendering and shading computations.

Functions:
    srgb_to_linear: Converts an sRGB texture to linear space.
    linear_to_srgb: Converts a linear texture to sRGB space.
"""

import math

import torch
import torch.nn.functional as F


def srgb_to_linear(texture: torch.Tensor) -> torch.Tensor:
    """
    Convert an sRGB texture to linear space.

    Args:
        texture (torch.Tensor): The sRGB texture of shape (..., C, H, W).

    Returns:
        torch.Tensor: The texture in linear space.
    """
    # Ensure the texture is within [0, 1] range
    texture = texture.clamp(0, 1)
    mask = texture <= 0.04045
    linear_texture = torch.zeros_like(texture)
    linear_texture[mask] = texture[mask] / 12.92
    linear_texture[~mask] = ((texture[~mask] + 0.055) / 1.055) ** 2.4
    return linear_texture.clamp(0, 1)


def linear_to_srgb(texture: torch.Tensor) -> torch.Tensor:
    """
    Convert a linear texture to sRGB space.

    Args:
        texture (torch.Tensor): The linear texture of shape (..., C, H, W).

    Returns:
        torch.Tensor: The texture in sRGB space.
    """
    # Ensure the texture is within [0, 1] range
    texture = texture.clamp(0, 1)
    mask = texture <= 0.0031308
    srgb_texture = torch.zeros_like(texture)
    srgb_texture[mask] = texture[mask] * 12.92
    srgb_texture[~mask] = 1.055 * torch.pow(texture[~mask], 1 / 2.4) - 0.055
    return srgb_texture.clamp(0, 1)


def rotate_normals(normal_map: torch.FloatTensor, angle: float) -> torch.FloatTensor:
    """
    Rotate the normals in the normal map by the given angle.

    Args:
        normal_map (torch.FloatTensor): The normal map tensor.
        angle (float): The rotation angle in degrees.

    Returns:
        torch.FloatTensor: The adjusted normal map.
    """
    # Convert angle to radians
    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation_matrix = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        device=normal_map.device,
        dtype=normal_map.dtype,
    )

    # Reshape normal_map to apply rotation
    c, h, w = normal_map.shape
    n_x = normal_map[0].view(-1)
    n_y = normal_map[1].view(-1)
    n_z = normal_map[2].view(-1)

    n_xy = torch.stack([n_x, n_y], dim=1)  # Shape: (H*W, 2)
    n_xy_rotated = n_xy @ rotation_matrix.T  # Rotate normals

    # Normalize the normals
    n_xyz_rotated = torch.stack([n_xy_rotated[:, 0], n_xy_rotated[:, 1], n_z], dim=1)
    n_xyz_rotated = F.normalize(n_xyz_rotated, dim=1)

    # Reshape back to original shape
    normal_map[0] = n_xyz_rotated[:, 0].view(h, w)
    normal_map[1] = n_xyz_rotated[:, 1].view(h, w)
    normal_map[2] = n_xyz_rotated[:, 2].view(h, w)

    return normal_map


def invert_normal(normals: torch.FloatTensor) -> torch.FloatTensor:
    """
    Invert the Y component of the normal map.

    Returns:
        torch.FloatTensor: The inverted normal map.
    """
    if normals is not None:
        normals[1] = 1 - normals[1]
    return normals
