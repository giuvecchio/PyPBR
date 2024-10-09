"""
pypbr.blending.functional

This module provides functional interfaces for blending PBR materials using various methodologies.

Functions:
    blend_materials: Blend two materials together using the specified method.
    blend_with_mask: Blend two materials using a provided mask.
    blend_on_height: Blend two materials based on their height maps.
    blend_on_properties: Blend two materials based on a specified property map.
    blend_with_gradient: Blend two materials using a linear gradient mask.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from ..material import MaterialBase


def blend_materials(
    material1: MaterialBase, material2: MaterialBase, method: str = "mask", **kwargs
) -> MaterialBase:
    """
    Blend two materials together using the specified method.

    Args:
        material1 (MaterialBase): The first material.
        material2 (MaterialBase): The second material.
        method (str): The blending method to use. Options are 'mask', 'height', 'properties', 'gradient'.
        **kwargs: Additional arguments specific to the blending method.

    Returns:
        MaterialBase: A new material resulting from blending material1 and material2.

    Raises:
        ValueError: If an unknown blending method is specified or required parameters are missing.
    """
    if method == "mask":
        mask: Optional[torch.FloatTensor] = kwargs.get("mask", None)
        if mask is None:
            raise ValueError("Mask must be provided for 'mask' blending method.")
        return blend_with_mask(material1, material2, mask)
    elif method == "height":
        blend_width: float = kwargs.get("blend_width", 0.1)
        return blend_on_height(material1, material2, blend_width)
    elif method == "properties":
        property_name: str = kwargs.get("property_name", "metallic")
        blend_width: float = kwargs.get("blend_width", 0.1)
        return blend_on_properties(material1, material2, property_name, blend_width)
    elif method == "gradient":
        direction: str = kwargs.get("direction", "horizontal")
        return blend_with_gradient(material1, material2, direction)
    else:
        raise ValueError(f"Unknown blending method: {method}")


def blend_with_mask(
    material1: MaterialBase, material2: MaterialBase, mask: torch.FloatTensor
) -> MaterialBase:
    """
    Blend two materials using the provided mask.

    Args:
        material1 (MaterialBase): The first material.
        material2 (MaterialBase): The second material.
        mask (torch.FloatTensor): The blending mask tensor, with values in [0, 1], shape [1, H, W] or [H, W].

    Returns:
        MaterialBase: A new material resulting from blending material1 and material2.
    """
    # Create a new material to store the blended result
    blended_material: MaterialBase = (
        material1.__class__()
    )  # Instantiate the same class as material1

    # Ensure mask has shape [1, H, W]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() != 3 or mask.size(0) != 1:
        raise ValueError("Mask must have shape [1, H, W] or [H, W].")

    # Get the set of all map names present in either material
    map_names = set(material1._maps.keys()).union(material2._maps.keys())

    for name in map_names:
        map1: Optional[torch.FloatTensor] = material1._maps.get(name, None)
        map2: Optional[torch.FloatTensor] = material2._maps.get(name, None)

        if map1 is None and map2 is None:
            blended_map: Optional[torch.FloatTensor] = None
        elif map1 is None:
            blended_map = map2
        elif map2 is None:
            blended_map = map1
        else:
            if name == "normal":
                # Blend normals correctly
                blended_map = _blend_normals(map1, map2, mask)
            else:
                # Standard blending
                blended_map = mask * map1 + (1 - mask) * map2

        setattr(blended_material, name, blended_map)

    # Copy other attributes
    blended_material.albedo_is_srgb = material1.albedo_is_srgb
    blended_material.device = material1.device

    return blended_material, mask


def _blend_normals(
    normal1: torch.FloatTensor, normal2: torch.FloatTensor, mask: torch.FloatTensor
) -> torch.FloatTensor:
    """
    Blend two normal maps using the provided mask.

    Args:
        normal1 (torch.FloatTensor): The first normal map tensor, with shape [3, H, W].
        normal2 (torch.FloatTensor): The second normal map tensor, with shape [3, H, W].
        mask (torch.FloatTensor): The blending mask tensor, with values in [0, 1], shape [1, H, W].

    Returns:
        torch.FloatTensor: The blended normal map tensor.
    """
    # Normalize normals to ensure they are unit vectors
    normal1_normalized: torch.FloatTensor = F.normalize(normal1, dim=0)
    normal2_normalized: torch.FloatTensor = F.normalize(normal2, dim=0)

    # Perform blending
    blended_normal: torch.FloatTensor = (
        mask * normal1_normalized + (1 - mask) * normal2_normalized
    )

    # Re-normalize the blended normals
    blended_normal_normalized: torch.FloatTensor = F.normalize(blended_normal, dim=0)

    return blended_normal_normalized


def blend_on_height(
    material1: MaterialBase,
    material2: MaterialBase,
    blend_width: float = 0.1,
    shift: float = 0.0,
) -> MaterialBase:
    """
    Blend two materials based on their height maps with an optional vertical shift.

    Args:
        material1 (MaterialBase): The first material.
        material2 (MaterialBase): The second material.
        blend_width (float, optional): Controls the sharpness of the blending transition.
                                       Smaller values result in sharper transitions.
                                       Defaults to 0.1.
        shift (float, optional): Shifts the blending transition vertically.
                                 Positive values raise the blending height,
                                 while negative values lower it.
                                 Defaults to 0.0.

    Returns:
        MaterialBase: A new material resulting from blending material1 and material2 based on height.

    Raises:
        ValueError: If either material lacks a height map.
    """
    height1: Optional[torch.FloatTensor] = material1._maps.get("height", None)
    height2: Optional[torch.FloatTensor] = material2._maps.get("height", None)

    if height1 is None or height2 is None:
        raise ValueError(
            "Both materials must have height maps for height-based blending."
        )

    # Ensure height maps are the same size
    if height1.shape != height2.shape:
        # Resize height2 to match height1's size
        height2 = TF.resize(height2, height1.shape[1:], antialias=True)

    # Apply the shift to material1's height map
    height1_shifted: torch.FloatTensor = height1 + shift

    # Compute the blend mask based on the shifted height difference
    height_diff: torch.FloatTensor = height1_shifted - height2

    # Apply a sigmoid function to create a smooth transition
    mask: torch.FloatTensor = torch.sigmoid(height_diff / (blend_width + 1e-6))

    return blend_with_mask(material1, material2, mask)


def blend_on_properties(
    material1: MaterialBase,
    material2: MaterialBase,
    property_name: str = "metallic",
    blend_width: float = 0.1,
) -> MaterialBase:
    """
    Blend two materials based on a specified property map.

    Args:
        material1 (MaterialBase): The first material.
        material2 (MaterialBase): The second material.
        property_name (str): The name of the property map to use for blending (e.g., 'metallic', 'roughness').
        blend_width (float): Controls the sharpness of the blending transition.

    Returns:
        MaterialBase: A new material resulting from blending material1 and material2.

    Raises:
        ValueError: If either material lacks the specified property map.
    """
    prop1: Optional[torch.FloatTensor] = material1._maps.get(property_name, None)
    prop2: Optional[torch.FloatTensor] = material2._maps.get(property_name, None)

    if prop1 is None or prop2 is None:
        raise ValueError(
            f"Both materials must have '{property_name}' maps for property-based blending."
        )

    # Ensure property maps are the same size
    if prop1.shape != prop2.shape:
        # Resize prop2 to match prop1's size
        prop2 = TF.resize(prop2, prop1.shape[1:], antialias=True)

    # Compute the blend mask based on the property difference
    prop_diff: torch.FloatTensor = prop1 - prop2

    # Apply a sigmoid function to create a smooth transition
    mask: torch.FloatTensor = torch.sigmoid(prop_diff / (blend_width + 1e-6))

    return blend_with_mask(material1, material2, mask)


def blend_with_gradient(
    material1: MaterialBase, material2: MaterialBase, direction: str = "horizontal"
) -> MaterialBase:
    """
    Blend two materials using a linear gradient mask.

    Args:
        material1 (MaterialBase): The first material.
        material2 (MaterialBase): The second material.
        direction (str): The direction of the gradient ('horizontal' or 'vertical').

    Returns:
        MaterialBase: A new material resulting from blending material1 and material2.

    Raises:
        ValueError: If an invalid direction is specified or material size cannot be determined.
    """
    # Get the size of the materials
    size: Optional[Tuple[int, int]] = material1.size
    if size is None:
        raise ValueError("Materials must have at least one map to determine size.")

    H, W = size
    device = material1.device

    if direction == "horizontal":
        gradient: torch.FloatTensor = (
            torch.linspace(0, 1, steps=W, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, H, 1)
        )
    elif direction == "vertical":
        gradient = (
            torch.linspace(0, 1, steps=H, device=device)
            .unsqueeze(0)
            .unsqueeze(2)
            .repeat(1, 1, W)
        )
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    mask: torch.FloatTensor = gradient

    return blend_with_mask(material1, material2, mask)
