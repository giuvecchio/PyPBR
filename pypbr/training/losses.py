"""
pypbr.training.losses

Defines loss functions for training PBR material reconstruction models.
This module includes composite loss functions that aggregate errors over multiple texture maps.
"""

import torch
import torch.nn as nn

from ..materials import MaterialBase
from ..models import CookTorranceBRDF


def material_reconstruction_loss(
    predicted: MaterialBase,
    target: MaterialBase,
    loss_fn=nn.L1Loss(),
    weights=None,
    return_dict: bool = False,
    average_loss: bool = True,
):
    """
    Compute a composite loss between two MaterialBase objects.

    This loss aggregates L1 losses for common maps (e.g., albedo, normal, roughness).

    Args:
        predicted (MaterialBase): Predicted material.
        target (MaterialBase): Ground truth material.
        loss_fn: A loss function (default: L1 loss).
        weights (dict, optional): Dictionary specifying weights for each map.
            Example: {'albedo': 1.0, 'normal': 1.0, 'roughness': 0.5}
        return_dict (bool, optional): Wether to return a dict of separate losses.
        average_loss (bool, optional): Wether to average the total loss by the number of maps.

    Returns:
        torch.Tensor: The aggregated loss.
        Optional[Dict]: Dict containing the loss for each map.
    """
    if weights is None:
        weights = {
            map_name: 1.0
            for map_name in predicted._maps.keys()
            if map_name in target._maps.keys()
        }

    maps_losses = {}
    for map_name, weight in weights.items():
        try:
            pred_map = getattr(predicted, map_name)
            target_map = getattr(target, map_name)
        except AttributeError:
            continue  # Skip maps that are not available in both materials
        if pred_map is not None and target_map is not None:
            maps_losses[map_name] = weight * loss_fn(pred_map, target_map)

    total_loss = sum(maps_losses.values())
    if average_loss:
        total_loss /= len(maps_losses.values())

    if return_dict:
        return total_loss, maps_losses

    return total_loss


def material_rendering_loss(
    predicted: MaterialBase,
    target: MaterialBase,
    loss_fn=nn.L1Loss(),
    light_dir: torch.Tensor = None,
    light_intensity: torch.Tensor = torch.tensor([1.0, 1.0, 1.0]),
    light_size: float = 1.0,
):
    """
    Compute a composite loss between the renderings of two MaterialBase objects.

    Args:
        predicted (MaterialBase): Predicted material.
        target (MaterialBase): Ground truth material.
        loss_fn: A loss function (default: L1 loss).
        light_dir: Light position.
        light_intensity: Strength and color of the light (default: white light).
        light_size: Size of the point light (default: 1.0).
    Returns:
        torch.Tensor: The aggregated loss.
    """

    # Create an instance of the BRDF
    brdf = CookTorranceBRDF(light_type="point")

    # Define the view direction, light direction, and light intensity
    view_dir = torch.tensor([0.0, 0.0, 1.0])  # Viewing straight on
    if light_dir == None:
        light_dir = torch.rand(3) + torch.tensor([0.0, 0.0, 1.0])

    # Evaluate the BRDF to get the reflected color
    render_pred = brdf(predicted, view_dir, light_dir, light_intensity, light_size)
    render_target = brdf(target, view_dir, light_dir, light_intensity, light_size)

    loss = loss_fn(render_pred, render_target)

    return loss
