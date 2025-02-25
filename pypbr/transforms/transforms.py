"""
pypbr.transforms.transforms

This module provides class-based interfaces for blending PBR materials using various methodologies.

Classes:
    `Compose`: Compose several transforms together.
    `Resize`: Resize all texture maps in the material to the specified size.
    `RandomResize`: Randomly resize all texture maps in the material to a size within a specified range.
    `Crop`: Crop all texture maps in the material to the specified region.
    `CenterCrop`: Crop all texture maps in the material to the center region.
    `RandomCrop`: Randomly crop all texture maps in the material to a specified size.
    `Tile`: Tile all texture maps in the material by repeating them.
    `Rotate`: Rotate all texture maps in the material by a specified angle.
    `RandomRotate`: Randomly rotate all texture maps in the material by a specified range of angles.
    `FlipHorizontal`: Flip all texture maps in the material horizontally.
    `FlipVertical`: Flip all texture maps in the material vertically.
    `RandomHorizontalFlip`: Randomly flip all texture maps in the material horizontally.
    `RandomVerticalFlip`: Randomly flip all texture maps in the material vertically.
    `Roll`: Roll all texture maps in the material along specified shift dimensions.
    `InvertNormal`: Invert the Y component of the normal map in the material.
    `AdjustNormalStrength`: Adjust the strength of the normal map in the material.
    `ToLinear`: Convert the albedo (and specular if present) maps to linear space.
    `ToSrgb`: Convert the albedo (and specular if present) maps to sRGB space.
"""

from typing import List, Tuple

from ..materials import MaterialBase
from . import functional


class Compose:
    """
    Compose several transforms together.

    Args:
        transforms (List[Callable]): A list of transform instances to be applied sequentially.
    """

    def __init__(self, transforms: List[callable]):
        self.transforms = transforms

    def __call__(self, material: MaterialBase) -> MaterialBase:
        """
        Apply each transform in sequence to the material.

        Args:
            material (MaterialBase): The input material.

        Returns:
            MaterialBase: The transformed material after all transformations.
        """
        for transform in self.transforms:
            material = transform(material)
        return material


# Resizing


class Resize:
    """
    Resize all texture maps in the material to the specified size.
    """

    def __init__(self, size: Tuple[int, int], antialias: bool = True):
        """
        Initialize the Resize transform.

        Args:
            size (Tuple[int, int]): Desired output size as (height, width).
            antialias (bool): Whether to apply antialiasing.
        """
        self.size = size
        self.antialias = antialias

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.resize(material, size=self.size, antialias=self.antialias)


class RandomResize:
    """
    Randomly resize all texture maps in the material to a size within a specified range.
    """

    def __init__(self, min_size: int, max_size: int, antialias: bool = True):
        """
        Initialize the RandomResize transform.

        Args:
            min_size (int): The minimum size for resizing.
            max_size (int): The maximum size for resizing.
            antialias (bool): Whether to apply antialiasing.
        """
        self.min_size = min_size
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.random_resize(
            material,
            min_size=self.min_size,
            max_size=self.max_size,
            antialias=self.antialias,
        )


# Cropping


class Crop:
    """
    Crop all texture maps in the material to the specified region.
    """

    def __init__(self, top: int, left: int, height: int, width: int):
        """
        Initialize the Crop transform.

        Args:
            top (int): The top pixel coordinate.
            left (int): The left pixel coordinate.
            height (int): The height of the crop.
            width (int): The width of the crop.
        """
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.crop(
            material,
            top=self.top,
            left=self.left,
            height=self.height,
            width=self.width,
        )


class CenterCrop:
    """
    Crop all texture maps in the material to the center region.
    """

    def __init__(self, height: int, width: int):
        """
        Initialize the CenterCrop transform.

        Args:
            height (int): The height of the crop.
            width (int): The width of the crop.
        """
        self.height = height
        self.width = width

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.center_crop(material, crop_size=(self.height, self.width))


class RandomCrop:
    """
    Randomly crop all texture maps in the material to a specified size.
    """

    def __init__(self, height: int, width: int):
        """
        Initialize the RandomCrop transform.

        Args:
            height (int): The height of the crop.
            width (int): The width of the crop.
        """
        self.height = height
        self.width = width

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.random_crop(material, crop_size=(self.height, self.width))


# Tiling/Padding


class Tile:
    """
    Tile all texture maps in the material by repeating them.
    """

    def __init__(self, num_tiles: int):
        """
        Initialize the Tile transform.

        Args:
            num_tiles (int): Number of times to tile the textures.
        """
        self.num_tiles = num_tiles

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.tile(material, num_tiles=self.num_tiles)


# Rotation


class Rotate:
    """
    Rotate all texture maps in the material by a specified angle.
    """

    def __init__(
        self,
        angle: float,
        expand: bool = False,
        padding_mode: str = "constant",
    ):
        """
        Initialize the Rotate transform.

        Args:
            angle (float): The rotation angle in degrees.
            expand (bool): Whether to expand the output image to hold the entire rotated image.
            padding_mode (str): Padding mode. Options are 'constant' or 'circular'.
        """
        assert padding_mode in ["constant", "circular"], "Invalid padding mode."
        self.angle = angle
        self.expand = expand
        self.padding_mode = padding_mode

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.rotate(
            material,
            angle=self.angle,
            expand=self.expand,
            padding_mode=self.padding_mode,
        )


class RandomRotate:
    """
    Randomly rotate all texture maps in the material by a specified range of angles.
    """

    def __init__(
        self,
        min_angle: float = 0.0,
        max_angle: float = 360.0,
        expand: bool = False,
        padding_mode: str = "constant",
    ):
        """
        Initialize the RandomRotate transform.

        Args:
            min_angle (float): The minimum rotation angle in degrees.
            max_angle (float): The maximum rotation angle in degrees.
            expand (bool): Whether to expand the output image to hold the entire rotated image.
            padding_mode (str): Padding mode. Options are 'constant' or 'circular'.
        """
        assert padding_mode in ["constant", "circular"], "Invalid padding mode."
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.expand = expand
        self.padding_mode = padding_mode

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.random_rotate(
            material,
            min_angle=self.min_angle,
            max_angle=self.max_angle,
            expand=self.expand,
            padding_mode=self.padding_mode,
        )


# Flipping


class FlipHorizontal:
    """
    Flip all texture maps in the material horizontally.
    """

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.flip_horizontal(material)


class FlipVertical:
    """
    Flip all texture maps in the material vertically.
    """

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.flip_vertical(material)


class RandomHorizontalFlip:
    """
    Randomly flip all texture maps in the material horizontally.
    """

    def __call__(self, material: MaterialBase, p: float = 0.5) -> MaterialBase:
        return functional.random_horizontal_flip(material, p)


class RandomVerticalFlip:
    """
    Randomly flip all texture maps in the material vertically.
    """

    def __call__(self, material: MaterialBase, p: float = 0.5) -> MaterialBase:
        return functional.random_vertical_flip(material, p)


# Translation


class Roll:
    """
    Roll all texture maps in the material along specified shift dimensions.
    """

    def __init__(self, shift: Tuple[int, int]):
        """
        Initialize the Roll transform.

        Args:
            shift (Tuple[int, int]): The shift values for (height, width) dimensions.
        """
        self.shift = shift

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.roll(material, shift=self.shift)


# Normal Map Adjustments


class InvertNormal:
    """
    Invert the Y component of the normal map in the material.
    """

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.invert_normal_map(material)


class AdjustNormalStrength:
    """
    Adjust the strength of the normal map in the material.
    """

    def __init__(self, strength_factor: float):
        """
        Initialize the AdjustNormalStrength transform.

        Args:
            strength_factor (float): The factor to adjust the strength of the normal map.
        """
        self.strength_factor = strength_factor

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.adjust_normal_strength(
            material, strength_factor=self.strength_factor
        )


# Color Space Conversion


class ToLinear:
    """
    Convert the albedo (and specular if present) maps to linear space.
    """

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.to_linear(material)


class ToSrgb:
    """
    Convert the albedo (and specular if present) maps to sRGB space.
    """

    def __call__(self, material: MaterialBase) -> MaterialBase:
        return functional.to_srgb(material)
