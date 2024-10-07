"""
pypbr.utils

This module provides utility functions for color space conversions between sRGB and linear space.
These functions are used in Physically Based Rendering (PBR) pipelines to ensure correct color 
representation during material rendering and shading computations.

Functions:
    srgb_to_linear: Converts an sRGB texture to linear space.
    linear_to_srgb: Converts a linear texture to sRGB space.
    rotate_normals: Rotates the normals in a normal map by a given angle.
    invert_normal: Inverts the Y component of the normal map.
    compute_normal_from_height: Computes the normal map from a height map.
    compute_height_from_normal: Computes the height map from a normal map.
"""

import math

import numpy as np
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
    Invert the Y component of the normal map. The normal map should be normalized.

    Returns:
        torch.FloatTensor: The inverted normal map.
    """
    if normals is not None:
        normals[1] = -normals[1]
    return normals


def compute_normal_from_height(
    height_map: torch.FloatTensor, scale: float = 1.0
) -> torch.FloatTensor:
    """
    Compute the normal map from the height map.

    Args:
        height_map (torch.FloatTensor): The height map tensor.
        scale (float): The scaling factor for the height map gradients.
                        Controls the strength of the normals.

    Returns:
        torch.FloatTensor: The normal map tensor.
    """
    if height_map is None:
        raise ValueError("Height map is required to compute normals.")

    if height_map.dim() == 2:
        height_map = height_map.unsqueeze(0)  # Add channel dimension

    # Compute gradients along X and Y axes
    grad_x = (
        F.pad(height_map, (1, 0, 0, 0))[:, :, :-1]
        - F.pad(height_map, (0, 1, 0, 0))[:, :, 1:]
    )
    grad_y = (
        F.pad(height_map, (0, 0, 1, 0))[:, :-1, :]
        - F.pad(height_map, (0, 0, 0, 1))[:, 1:, :]
    )

    # Adjust gradients based on scale
    grad_x = grad_x * scale
    grad_y = grad_y * scale

    # Create normal map components
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = torch.ones_like(height_map)

    # Concatenate components and normalize
    normal_map = torch.cat([normal_x, normal_y, normal_z], dim=0)
    normal_map = F.normalize(normal_map, dim=0)

    return normal_map


def compute_height_from_normal(
    normal_map: torch.FloatTensor, scale: float = 1.0
) -> torch.FloatTensor:
    """
    Compute the height map from the normal map using Poisson reconstruction.

    Args:
        scale (float): Scaling factor for the gradients.

    Returns:
        MaterialBase: Returns self for method chaining.
    """
    if normal_map is None:
        raise ValueError("Normal map is required to compute height.")

    # Ensure normal map has three channels (C, H, W)
    if normal_map.shape[0] != 3:
        raise ValueError("Normal map must have three channels.")

    # Get the X, Y, and Z components
    N_x = normal_map[0]
    N_y = normal_map[1]
    N_z = normal_map[2]

    # Avoid division by zero
    N_z = N_z + 1e-8

    # Compute gradients
    g_x = -N_x / N_z
    g_y = N_y / N_z

    # Scale the gradients
    g_x = g_x * scale
    g_y = g_y * scale

    # Compute divergence of the gradient field
    div_g = _compute_divergence(g_x, g_y)

    # Solve Poisson equation using FFT
    height_map = _poisson_solver(div_g)

    # Normalize the height map
    height_map = height_map - height_map.mean()

    # Normalize the height map to range [0, 1]
    height_min = height_map.min()
    height_max = height_map.max()

    height_range = height_max - height_min + 1e-8  # Avoid division by zero

    height_map = (height_map - height_min) / height_range

    height_map = height_map.unsqueeze(0)

    # Return the height map
    return height_map


def _compute_divergence(
    g_x: torch.FloatTensor, g_y: torch.FloatTensor
) -> torch.FloatTensor:
    """
    Compute the divergence of the gradient field.

    Args:
        g_x (torch.Tensor): Gradient in the x-direction.
        g_y (torch.Tensor): Gradient in the y-direction.

    Returns:
        torch.Tensor: The divergence of the gradient field.
    """
    # Add batch and channel dimensions
    g_x = g_x.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    g_y = g_y.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

    # Pad g_x on the right
    g_x_pad = F.pad(g_x, pad=(0, 1, 0, 0), mode="replicate")
    # Compute dgx_dx
    dgx_dx = g_x_pad[:, :, :, 1:] - g_x_pad[:, :, :, :-1]  # Shape: [1, 1, H, W]

    # Pad g_y on the bottom
    g_y_pad = F.pad(g_y, pad=(0, 0, 0, 1), mode="replicate")
    # Compute dgy_dy
    dgy_dy = g_y_pad[:, :, 1:, :] - g_y_pad[:, :, :-1, :]  # Shape: [1, 1, H, W]

    # Sum the derivatives to get divergence
    div_g = dgx_dx + dgy_dy  # Shape: [1, 1, H, W]

    # Remove batch and channel dimensions
    div_g = div_g.squeeze(0).squeeze(0)  # Shape: [H, W]

    return div_g


def _poisson_solver(div_g: torch.FloatTensor) -> torch.FloatTensor:
    """
    Solve the Poisson equation using the Fast Fourier Transform (FFT) method.

    Args:
        div_g (torch.Tensor): The divergence of the gradient field.

    Returns:
        torch.Tensor: The reconstructed height map.
    """
    # Get the height and width
    H, W = div_g.shape

    # Create frequency grids
    y = torch.arange(0, H, dtype=torch.float32, device=div_g.device).view(-1, 1)
    x = torch.arange(0, W, dtype=torch.float32, device=div_g.device).view(1, -1)

    # Convert to radians per pixel
    pi = np.pi
    yy = 2 * pi * y / H
    xx = 2 * pi * x / W

    # Compute eigenvalues of the Laplacian in the frequency domain
    denom = (2 * torch.cos(xx) - 2) + (2 * torch.cos(yy) - 2)
    denom[0, 0] = 1.0  # Avoid division by zero at the zero frequency

    # Perform FFT of the divergence
    div_g_fft = torch.fft.fft2(div_g)

    # Solve in frequency domain
    height_map_fft = div_g_fft / denom

    height_map_fft[0, 0] = 0  # Set the mean to zero (eliminate the DC component)

    # Inverse FFT to get the height map
    height_map = torch.fft.ifft2(height_map_fft).real

    return height_map
