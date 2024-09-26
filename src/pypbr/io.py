"""
pypbr.io

This module provides input/output utility functions for loading and saving material maps 
from and to folders, used in Physically Based Rendering (PBR) workflows. It handles image 
formats and naming conventions for various material properties such as base color, normal maps, 
roughness, height, and metallic maps.

Functions:
    load_material_from_folder: Loads material maps from a folder based on file naming conventions.
    save_material_to_folder: Saves material maps to a specified folder in the desired format.
"""

import os
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from typing import Optional, Dict
from .material import Material


def load_material_from_folder(
    folder_path: str,
    map_names: Optional[Dict[str, list]] = None,
) -> Material:
    """
    Load material maps from a folder using naming conventions.

    Args:
        folder_path: Path to the folder containing material maps.
        map_names: Optional dictionary specifying map types and their possible filenames.

    Returns:
        Material: An instance of the Material class with loaded maps.
    """
    if map_names is None:
        map_names = {
            "basecolor": ["basecolor", "albedo"],
            "normal": ["normal", "normalmap"],
            "height": ["height", "displacement"],
            "roughness": ["roughness"],
            "metallic": ["metallic", "metalness"],
        }

    material_kwargs = {}

    for map_type, possible_names in map_names.items():
        for name in possible_names:
            for ext in ["png", "jpg", "jpeg", "tiff", "bmp", "exr"]:
                filename = f"{name}.{ext}"
                filepath = os.path.join(folder_path, filename)
                if os.path.isfile(filepath):
                    image = Image.open(filepath)
                    if map_type in ["basecolor", "normal"]:
                        image = image.convert("RGB")
                    else:
                        image = image.convert("L") if image.mode != "L" else image
                    material_kwargs[map_type] = image
                    break
            if map_type in material_kwargs:
                break

    return Material(**material_kwargs)


def save_material_to_folder(
    material: Material,
    folder_path: str,
    map_names: Optional[Dict[str, str]] = None,
    format: str = "png",
):
    """
    Save material maps to a folder using naming conventions.

    Args:
        material: The material to save.
        folder_path: Path to the folder where maps will be saved.
        map_names: Optional dictionary specifying filenames for each map type.
        format: The image format to save the maps (e.g., 'png', 'jpg').
    """
    os.makedirs(folder_path, exist_ok=True)

    # Get maps as images
    as_pil = material.to_pil()

    # Save each map as an image
    for map_type, image in as_pil.items():
        filename = f"{map_names.get(map_type, map_type)}.{format}"
        filepath = os.path.join(folder_path, filename)
        image.save(filepath)
