"""
pypbr.transforms.functional

This module provides a set of functional transformations for manipulating texture maps in Physically Based Rendering (PBR) materials. Each transformation operates on a material instance and applies geometric, color space, or normal map adjustments to its texture maps (e.g., albedo, normal, roughness). These transformations can be used for rendering or preprocessing materials for machine learning tasks.

Functions:   
    `resize`: Resize all texture maps in a material to a specified size.

    `random_resize`: Randomly resize all texture maps in a material to a size within a specified range.
    
    `crop`: Crop all texture maps in a material to a specified region.

    `center_crop`: Center crop all texture maps in a material to a specified size.

    `random_crop`: Randomly crop all texture maps in a material to a specified size.
    
    `tile`: Tile all texture maps by repeating them a given number of times.
    
    `rotate`: Rotate texture maps in the material by a specified angle.

    `random_rotate`: Randomly rotate texture maps by a specified range of angles.
    
    `flip_horizontal`: Flip texture maps horizontally.
    
    `flip_vertical`: Flip texture maps vertically.

    `random_horizontal_flip`: Randomly flip texture maps horizontally based on a probability.

    `random_vertical_flip`: Randomly flip texture maps vertically based on a probability.
    
    `random_flip`: Randomly flip texture maps horizontally and/or vertically based on a probability.
    
    `roll`: Roll texture maps along the specified axes.
    
    `invert_normal_map`: Invert the Y component of the normal map.
    
    `adjust_normal_strength`: Adjust the strength of the normal map.
    
    `to_linear`: Convert albedo and specular maps to linear color space.
    
    `to_srgb`: Convert albedo and specular maps to sRGB color space.
"""

from random import random
from typing import Tuple

from ..materials import MaterialBase

# Resizing


def resize(
    material: MaterialBase, size: Tuple[int, int], antialias: bool = True
) -> MaterialBase:
    """
    Resize all texture maps in the material to the specified size.

    Args:
        material (MaterialBase): The input material.
        size (Tuple[int, int]): Desired output size as (height, width).
        antialias (bool): Whether to apply antialiasing.

    Returns:
        MaterialBase: A new material instance with resized maps.
    """
    new_material = material.clone()
    new_material.resize(size=size, antialias=antialias)
    return new_material


def random_resize(
    material: MaterialBase, min_size: int, max_size: int, antialias: bool = True
) -> MaterialBase:
    """
    Randomly resize all texture maps in the material to a size within a specified range.

    Args:
        material (MaterialBase): The input material.
        min_size (int): The minimum size for resizing.
        max_size (int): The maximum size for resizing.
        antialias (bool): Whether to apply antialiasing.

    Returns:
        MaterialBase: A new material instance with randomly resized maps.
    """
    new_material = material.clone()
    height = int(min_size + (max_size - min_size) * random())
    width = int(min_size + (max_size - min_size) * random())
    new_material.resize(size=(height, width), antialias=antialias)
    return new_material


# Cropping


def crop(
    material: MaterialBase, top: int, left: int, height: int, width: int
) -> MaterialBase:
    """
    Crop all texture maps in the material to the specified region.

    Args:
        material (MaterialBase): The input material.
        top (int): The top pixel coordinate.
        left (int): The left pixel coordinate.
        height (int): The height of the crop.
        width (int): The width of the crop.

    Returns:
        MaterialBase: A new material instance with cropped maps.
    """
    new_material = material.clone()
    new_material.crop(top=top, left=left, height=height, width=width)
    return new_material


def center_crop(material: MaterialBase, crop_size: Tuple[int, int]) -> MaterialBase:
    """
    Center crop all texture maps in the material to a specified size.

    Args:
        material (MaterialBase): The input material.
        crop_size (Tuple[int, int]): The desired crop size as (height, width).

    Returns:
        MaterialBase: A new material instance with center cropped maps.
    """
    new_material = material.clone()
    height, width = new_material.size
    crop_height, crop_width = crop_size
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    new_material.crop(top=top, left=left, height=crop_height, width=crop_width)
    return new_material


def random_crop(material: MaterialBase, crop_size: Tuple[int, int]) -> MaterialBase:
    """
    Randomly crop all texture maps in the material to a specified size.

    Args:
        material (MaterialBase): The input material.
        crop_size (Tuple[int, int]): The desired crop size as (height, width).

    Returns:
        MaterialBase: A new material instance with randomly cropped maps.
    """
    new_material = material.clone()
    height, width = new_material.size
    crop_height, crop_width = crop_size
    top = int((height - crop_height) * random())
    left = int((width - crop_width) * random())
    new_material.crop(top=top, left=left, height=crop_height, width=crop_width)
    return new_material


# Tiling/Padding


def tile(material: MaterialBase, num_tiles: int) -> MaterialBase:
    """
    Tile all texture maps in the material by repeating them.

    Args:
        material (MaterialBase): The input material.
        num_tiles (int): Number of times to tile the textures.

    Returns:
        MaterialBase: A new material instance with tiled maps.
    """
    new_material = material.clone()
    new_material.tile(num_tiles=num_tiles)
    return new_material


# Rotation


def rotate(
    material: MaterialBase,
    angle: float,
    expand: bool = False,
    padding_mode: str = "constant",
) -> MaterialBase:
    """
    Rotate all texture maps in the material by a specified angle.

    Args:
        material (MaterialBase): The input material.
        angle (float): The rotation angle in degrees.
        expand (bool): Whether to expand the output image to hold the entire rotated image.
        padding_mode (str): Padding mode. Options are 'constant' or 'circular'.

    Returns:
        MaterialBase: A new material instance with rotated maps.
    """
    new_material = material.clone()
    new_material.rotate(angle=angle, expand=expand, padding_mode=padding_mode)
    return new_material


def random_rotate(
    material: MaterialBase,
    min_angle: float = 0.0,
    max_angle: float = 360.0,
    expand: bool = False,
    padding_mode: str = "constant",
) -> MaterialBase:
    """
    Randomly rotate all texture maps in the material by a specified range of angles.

    Args:
        material (MaterialBase): The input material.
        min_angle (float): The minimum rotation angle in degrees.
        max_angle (float): The maximum rotation angle in degrees.
        expand (bool): Whether to expand the output image to hold the entire rotated image.
        padding_mode (str): Padding mode. Options are 'constant' or 'circular'.

    Returns:
        MaterialBase: A new material instance with randomly rotated maps.
    """
    new_material = material.clone()
    angle = min_angle + (max_angle - min_angle) * random()
    new_material.rotate(angle=angle, expand=expand, padding_mode=padding_mode)
    return new_material


# Flipping


def flip_horizontal(material: MaterialBase) -> MaterialBase:
    """
    Flip all texture maps in the material horizontally.

    Args:
        material (MaterialBase): The input material.

    Returns:
        MaterialBase: A new material instance with horizontally flipped maps.
    """
    new_material = material.clone()
    new_material.flip_horizontal()
    return new_material


def flip_vertical(material: MaterialBase) -> MaterialBase:
    """
    Flip all texture maps in the material vertically.

    Args:
        material (MaterialBase): The input material.

    Returns:
        MaterialBase: A new material instance with vertically flipped maps.
    """
    new_material = material.clone()
    new_material.flip_vertical()
    return new_material


def random_horizontal_flip(material: MaterialBase, p: float = 0.5) -> MaterialBase:
    """
    Randomly flip all texture maps in the material horizontally based on a probability.

    Args:
        material (MaterialBase): The input material.
        p (float): The probability of flipping the maps horizontally.

    Returns:
        MaterialBase: A new material instance with randomly flipped maps.
    """
    new_material = material.clone()
    if random() < p:
        new_material.flip_horizontal()
    return new_material


def random_vertical_flip(material: MaterialBase, p: float = 0.5) -> MaterialBase:
    """
    Randomly flip all texture maps in the material vertically based on a probability.

    Args:
        material (MaterialBase): The input material.
        p (float): The probability of flipping the maps vertically.

    Returns:
        MaterialBase: A new material instance with randomly flipped maps.
    """
    new_material = material.clone()
    if random() < p:
        new_material.flip_vertical()
    return new_material


# Translation


def roll(material: MaterialBase, shift: Tuple[int, int]) -> MaterialBase:
    """
    Roll all texture maps in the material along specified shift dimensions.

    Args:
        material (MaterialBase): The input material.
        shift (Tuple[int, int]): The shift values for (height, width) dimensions.

    Returns:
        MaterialBase: A new material instance with rolled maps.
    """
    new_material = material.clone()
    new_material.roll(shift=shift)
    return new_material


# Normal Map Adjustments


def invert_normal_map(material: MaterialBase) -> MaterialBase:
    """
    Invert the Y component of the normal map in the material.

    Args:
        material (MaterialBase): The input material.

    Returns:
        MaterialBase: A new material instance with inverted normal map.
    """
    new_material = material.clone()
    new_material.invert_normal()
    return new_material


def adjust_normal_strength(
    material: MaterialBase, strength_factor: float
) -> MaterialBase:
    """
    Adjust the strength of the normal map in the material.

    Args:
        material (MaterialBase): The input material.
        strength_factor (float): The factor to adjust the strength of the normal map.

    Returns:
        MaterialBase: A new material instance with adjusted normal strength.
    """
    new_material = material.clone()
    new_material.adjust_normal_strength(strength_factor=strength_factor)
    return new_material


# Color Space Transformations


def to_linear(material: MaterialBase) -> MaterialBase:
    """
    Convert the albedo (and specular if present) maps to linear space.

    Args:
        material (MaterialBase): The input material.

    Returns:
        MaterialBase: A new material instance with linear albedo and specular maps.
    """
    new_material = material.clone()
    new_material.to_linear()
    return new_material


def to_srgb(material: MaterialBase) -> MaterialBase:
    """
    Convert the albedo (and specular if present) maps to sRGB space.

    Args:
        material (MaterialBase): The input material.

    Returns:
        MaterialBase: A new material instance with sRGB albedo and specular maps.
    """
    new_material = material.clone()
    new_material.to_srgb()
    return new_material
