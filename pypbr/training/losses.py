"""
pypbr.training.losses

Defines loss functions for training PBR material reconstruction models.
This module includes composite loss functions that aggregate errors over multiple texture maps.
"""

import torch.nn as nn


def material_reconstruction_loss(predicted, target, loss_fn=nn.L1Loss(), weights=None):
    """
    Compute a composite loss between two MaterialBase objects.

    This loss aggregates L1 losses for common maps (e.g., albedo, normal, roughness).

    Args:
        predicted (MaterialBase): Predicted material.
        target (MaterialBase): Ground truth material.
        loss_fn: A loss function (default: L1 loss).
        weights (dict, optional): Dictionary specifying weights for each map.
            Example: {'albedo': 1.0, 'normal': 1.0, 'roughness': 0.5}

    Returns:
        torch.Tensor: The aggregated loss.
    """
    if weights is None:
        weights = {"albedo": 1.0, "normal": 1.0, "roughness": 1.0}

    total_loss = 0.0
    for map_name, weight in weights.items():
        try:
            pred_map = getattr(predicted, map_name)
            target_map = getattr(target, map_name)
        except AttributeError:
            continue  # Skip maps that are not available in both materials
        if pred_map is not None and target_map is not None:
            total_loss += weight * loss_fn(pred_map, target_map)
    return total_loss
