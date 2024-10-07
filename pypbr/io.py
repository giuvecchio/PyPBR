"""
pypbr.io

This module provides input/output utility functions for loading and saving material maps
from and to folders, used in Physically Based Rendering (PBR) workflows. It handles image
formats and naming conventions for various material properties such as albedo, normal maps,
roughness, height, metallic, and specular maps.

Functions:
    load_material_from_folder: Loads material maps from a folder based on file naming conventions.
    save_material_to_folder: Saves material maps to a specified folder in the desired format.
"""

import os
import warnings
from PIL import Image
from typing import Optional, Dict, List, Type

from .material import (
    MaterialBase,
    BasecolorMetallicMaterial,
    DiffuseSpecularMaterial,
)


def load_material_from_folder(
    folder_path: str,
    map_names: Optional[Dict[str, List[str]]] = None,
    preferred_workflow: Optional[str] = None,
    is_srgb: bool = True,
) -> MaterialBase:
    """
    Load material maps from a folder using naming conventions.

    Args:
        folder_path (str): Path to the folder containing material maps.
        map_names (Optional[Dict[str, List[str]]]): Optional dictionary specifying map types and their possible filenames.
        preferred_workflow (Optional[str]): Preferred workflow ('metallic' or 'specular'). If not specified, the workflow is determined based on available maps.
        is_srgb (bool): Whether the albedo and specular maps are in sRGB color space.

    Returns:
        MaterialBase: An instance of a Material subclass with loaded maps.
    """
    if map_names is None:
        map_names = {
            "basecolor": ["albedo", "basecolor"],
            "diffuse": ["diffuse"],
            "normal": ["normal", "normalmap"],
            "height": ["height", "displacement", "bump"],
            "roughness": ["roughness"],
            "metallic": ["metallic", "metalness"],
            "specular": ["specular"],
            # Additional maps can be added here
        }

    supported_extensions = ["png", "jpg", "jpeg", "tiff", "bmp", "exr"]

    # Dictionary to hold loaded images
    loaded_maps = {}

    # Scan for maps in the folder
    for map_type, possible_names in map_names.items():
        for name in possible_names:
            for ext in supported_extensions:
                filename = f"{name}.{ext}"
                filepath = os.path.join(folder_path, filename)
                if os.path.isfile(filepath):
                    image = Image.open(filepath)
                    if map_type in ["basecolor", "diffuse", "normal", "specular"]:
                        image = image.convert("RGB")
                    elif map_type == "height":
                        # For height maps, preserve the original mode if it's 16-bit or 32-bit
                        if image.mode in ["I", "I;16", "I;16B", "I;16L", "I;16N"]:
                            # Image is 16-bit unsigned integer
                            pass  # Keep original mode
                        elif image.mode == "F":
                            # Image is 32-bit floating point
                            pass  # Keep original mode
                        else:
                            # Ensure height map is in grayscale mode if not 16-bit or 32-bit
                            image = image.convert("L")

                    else:
                        image = image.convert("L") if image.mode != "L" else image
                    loaded_maps[map_type] = image
                    break  # Stop searching for this map_type once found
            if map_type in loaded_maps:
                break  # Stop searching other possible names once found

    # Determine which material class to instantiate
    material_class = select_material_class(loaded_maps, preferred_workflow)

    # Decide which albedo map to use based on the selected material class
    if issubclass(material_class, BasecolorMetallicMaterial):
        # Use 'basecolor' as albedo map
        albedo_map = loaded_maps.get("basecolor", None)
        if albedo_map is None:
            warnings.warn(
                "Basecolor map not found for metallic workflow. Looking for 'albedo' or 'basecolor' maps."
            )
        # Remove 'diffuse' map if present
        loaded_maps.pop("diffuse", None)
    elif issubclass(material_class, DiffuseSpecularMaterial):
        # Use 'diffuse' as albedo map
        albedo_map = loaded_maps.get("diffuse", None)
        if albedo_map is None:
            warnings.warn(
                "Diffuse map not found for specular workflow. Looking for 'diffuse' map."
            )
        # Remove 'basecolor' map if present
        loaded_maps.pop("basecolor", None)
    else:
        albedo_map = None

    # Prepare kwargs for material class instantiation
    material_kwargs = {
        k: v for k, v in loaded_maps.items() if k not in ["basecolor", "diffuse"]
    }
    material_kwargs["albedo"] = albedo_map

    # Create the material instance
    material_instance = material_class(
        **material_kwargs,
        albedo_is_srgb=is_srgb,  # Assuming albedo is in sRGB space
        specular_is_srgb=is_srgb,  # Assuming specular map is in sRGB space
    )

    return material_instance


def select_material_class(
    loaded_maps: Dict[str, Image.Image],
    preferred_workflow: Optional[str] = None,
) -> Type[MaterialBase]:
    """
    Select the appropriate Material subclass based on the loaded maps.

    Args:
        loaded_maps (Dict[str, Image.Image]): Dictionary of loaded maps.
        preferred_workflow (Optional[str]): Preferred workflow ('metallic' or 'specular').

    Returns:
        Type[MaterialBase]: The Material subclass to instantiate.
    """
    has_metallic = "metallic" in loaded_maps
    has_specular = "specular" in loaded_maps

    if has_metallic and has_specular:
        if preferred_workflow == "metallic":
            warnings.warn(
                "Both metallic and specular maps are present. Using metallic workflow as preferred."
            )
            # Remove 'specular' map since we're using metallic workflow
            loaded_maps.pop("specular", None)
            return BasecolorMetallicMaterial
        elif preferred_workflow == "specular":
            warnings.warn(
                "Both metallic and specular maps are present. Using specular workflow as preferred."
            )
            # Remove 'metallic' map since we're using specular workflow
            loaded_maps.pop("metallic", None)
            return DiffuseSpecularMaterial
        else:
            warnings.warn(
                "Both metallic and specular maps are present. Specify preferred_workflow to choose. Defaulting to metallic workflow."
            )
            # Default to metallic workflow
            loaded_maps.pop("specular", None)
            return BasecolorMetallicMaterial
    elif has_metallic:
        return BasecolorMetallicMaterial
    elif has_specular:
        return DiffuseSpecularMaterial
    else:
        # Decide based on available albedo maps
        if "basecolor" in loaded_maps:
            return BasecolorMetallicMaterial
        elif "diffuse" in loaded_maps:
            return DiffuseSpecularMaterial
        else:
            # Default to BasecolorMetallicMaterial and issue a warning
            warnings.warn(
                "Neither metallic nor specular map found, and no albedo map found. Defaulting to BasecolorMetallicMaterial."
            )
            return BasecolorMetallicMaterial


def save_material_to_folder(
    material: MaterialBase,
    folder_path: str,
    map_names: Optional[Dict[str, str]] = None,
    format: str = "png",
):
    """
    Save material maps to a folder using naming conventions.

    Args:
        material (MaterialBase): The material to save.
        folder_path (str): Path to the folder where maps will be saved.
        map_names (Optional[Dict[str, str]]): Optional dictionary specifying filenames for each map type.
        format (str): The image format to save the maps (e.g., 'png', 'jpg').
    """
    os.makedirs(folder_path, exist_ok=True)

    # Default map names if not provided
    if map_names is None:
        map_names = {
            "albedo": "albedo",
            "normal": "normal",
            "height": "height",
            "roughness": "roughness",
            "metallic": "metallic",
            "specular": "specular",
            # Additional maps can be added here
        }

    # Get maps as images
    as_pil = material.to_pil()

    # Save each map as an image
    for map_type, image in as_pil.items():
        if image is not None:
            # Remove leading underscore if present
            map_type_clean = map_type.lstrip("_")
            filename = f"{map_names.get(map_type_clean, map_type_clean)}.{format}"
            filepath = os.path.join(folder_path, filename)
            image.save(filepath)
        else:
            continue  # Skip saving if image is None
