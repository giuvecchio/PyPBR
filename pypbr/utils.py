"""
pypbr.utils

This module provides utility functions for color space conversions between sRGB and linear space.
These functions are used in Physically Based Rendering (PBR) pipelines to ensure correct color 
representation during material rendering and shading computations.

Functions:
    srgb_to_linear: Converts an sRGB texture to linear space.
    linear_to_srgb: Converts a linear texture to sRGB space.
"""

import torch


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
