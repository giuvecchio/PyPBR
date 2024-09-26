"""
pypbr.material

This module defines the Material class, which encapsulates various texture maps
used in Physically Based Rendering (PBR). It provides functionalities to manipulate
and convert these texture maps for rendering purposes.

Classes:
    Material: Represents a PBR material with attributes for different texture maps.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from .utils import linear_to_srgb, srgb_to_linear


class Material:
    """
    A class representing a PBR material.

    Attributes:
        basecolor (torch.FloatTensor): The basecolor map tensor.
        basecolor_is_srgb (bool): Flag indicating if basecolor is in sRGB space.
        normal (torch.FloatTensor): The normal map tensor.
        height (torch.FloatTensor): The height map tensor.
        roughness (torch.FloatTensor): The roughness map tensor.
        metallic (torch.FloatTensor): The metallic map tensor.
    """

    def __init__(
        self,
        basecolor: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        basecolor_is_srgb: bool = True,
        normal: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        height: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        roughness: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        metallic: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
    ):
        """
        Initialize the Material with optional texture maps.

        Args:
            basecolor: The basecolor map.
            basecolor_is_srgb: Flag indicating if basecolor is in sRGB space.
            normal: The normal map.
            height: The height map.
            roughness: The roughness map.
            metallic: The metallic map.
        """
        self._basecolor = None
        self._normal = None
        self._height = None
        self._roughness = None
        self._metallic = None
        self._basecolor_is_srgb = basecolor_is_srgb

        self.basecolor = basecolor
        self.normal = normal
        self.height = height
        self.roughness = roughness
        self.metallic = metallic

    @property
    def basecolor(self):
        return self._basecolor

    @basecolor.setter
    def basecolor(self, value):
        self._basecolor = self._to_tensor(value)

    @property
    def linear_basecolor(self):
        if self._basecolor_is_srgb:
            return srgb_to_linear(self._basecolor)
        return self._basecolor

    @property
    def basecolor_is_srgb(self):
        return self._basecolor_is_srgb

    @basecolor_is_srgb.setter
    def basecolor_is_srgb(self, value):
        self._basecolor_is_srgb = value

    @property
    def normal(self):
        return self._normal

    @normal.setter
    def normal(self, value):
        self._normal = self._process_normal_map(self._to_tensor(value))

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = self._to_tensor(value)

    @property
    def roughness(self):
        return self._roughness

    @roughness.setter
    def roughness(self, value):
        self._roughness = self._to_tensor(value)

    @property
    def metallic(self):
        return self._metallic

    @metallic.setter
    def metallic(self, value):
        self._metallic = self._to_tensor(value)

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """
        Get the size of the texture maps.

        Returns:
            A tuple (height, width) representing the size of the texture maps.
            If multiple maps are present, returns the size of the first non-None map.
            Returns None if no maps are available.
        """
        for attr in ["_basecolor", "_normal", "_height", "_roughness", "_metallic"]:
            map_value = getattr(self, attr)
            if map_value is not None:
                _, height, width = map_value.shape
                return (height, width)
        return None

    def _process_normal_map(
        self, normal_map: Optional[torch.FloatTensor]
    ) -> Optional[torch.FloatTensor]:
        """
        Process the normal map by computing the Z-component if necessary and normalizing.

        Args:
            normal_map: The normal map tensor.

        Returns:
            torch.FloatTensor: The processed normal map.
        """
        if normal_map is None:
            return None

        if normal_map.shape[0] == 2:
            # Compute the Z-component
            normal_map = self._compute_normal_map_z_component(normal_map)
        elif normal_map.shape[0] == 3:
            # Convert from [0,1] to [-1,1]
            normal_map = normal_map * 2.0 - 1.0
            normal_map = F.normalize(normal_map, dim=0)
        else:
            raise ValueError("Normal map must have 2 or 3 channels.")

        return normal_map

    def _compute_normal_map_z_component(
        self, normal_xy: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Compute the Z-component of the normal map from the X and Y components.

        Args:
            normal_xy: A tensor containing the X and Y components of the normal map.

        Returns:
            torch.FloatTensor: The normal map tensor with X, Y, and Z components.
        """
        normal_xy = normal_xy * 2 - 1  # Scale from [0,1] to [-1,1]
        x = normal_xy[0:1]
        y = normal_xy[1:2]
        squared = x**2 + y**2
        z = torch.sqrt(torch.clamp(1.0 - squared, min=0.0))
        normal = torch.cat([x, y, z], dim=0)
        normal = F.normalize(normal, dim=0)
        return normal

    def resize(self, size: Union[int, Tuple[int, int]], antialias: bool = True):
        """
        Resize all texture maps to the specified size.

        Args:
            size: The desired output size.
            antialias: Whether to apply antialiasing.

        Returns:
            Material: Returns self for method chaining.
        """
        for attr in ["_basecolor", "_normal", "_height", "_roughness", "_metallic"]:
            map_value = getattr(self, attr)
            if map_value is not None:
                setattr(self, attr, TF.resize(map_value, size, antialias=antialias))
        return self

    def crop(self, top: int, left: int, height: int, width: int):
        """
        Crop all texture maps to the specified region.

        Args:
            top: The top pixel coordinate.
            left: The left pixel coordinate.
            height: The height of the crop.
            width: The width of the crop.

        Returns:
            Material: Returns self for method chaining.
        """
        for attr in ["_basecolor", "_normal", "_height", "_roughness", "_metallic"]:
            map_value = getattr(self, attr)
            if map_value is not None:
                cropped_map = TF.crop(map_value, top, left, height, width)
                setattr(self, attr, cropped_map)
        return self

    def tile(self, num_tiles: int):
        """
        Tile all texture maps by repeating them.

        Args:
            num_tiles: Number of times to tile the textures.

        Returns:
            Material: Returns self for method chaining.
        """
        for attr in ["_basecolor", "_normal", "_height", "_roughness", "_metallic"]:
            texture = getattr(self, attr)
            if texture is not None:
                setattr(self, attr, texture.repeat(1, num_tiles, num_tiles))
        return self

    def invert_normal(self):
        """
        Invert the Y component of the normal map.

        Returns:
            Material: Returns self for method chaining.
        """
        if self._normal is not None:
            self._normal[1] = 1 - self._normal[1]
        return self

    def apply_transform(self, transform):
        """
        Apply a transformation to all texture maps.

        Args:
            transform: A function that takes a tensor and returns a tensor.

        Returns:
            Material: Returns self for method chaining.
        """
        for attr in ["_basecolor", "_normal", "_height", "_roughness", "_metallic"]:
            map_value = getattr(self, attr)
            if map_value is not None:
                setattr(self, attr, transform(map_value))
        return self

    def to_linear(self):
        """
        Convert the basecolor map to linear space if it's in sRGB.

        Returns:
            Material: Returns self for method chaining.
        """
        if self._basecolor is not None and self.basecolor_is_srgb:
            self._basecolor = srgb_to_linear(self._basecolor)
            self.basecolor_is_srgb = False
        return self

    def to_srgb(self):
        """
        Convert the basecolor map to sRGB space if it's in linear space.

        Returns:
            Material: Returns self for method chaining.
        """
        if self._basecolor is not None and not self.basecolor_is_srgb:
            self._basecolor = linear_to_srgb(self._basecolor)
            self.basecolor_is_srgb = True
        return self

    def to_numpy(self):
        """
        Convert all texture maps to NumPy arrays.

        Returns:
            dict: A dictionary containing NumPy arrays of the texture maps.
        """
        maps = {}
        for attr in ["basecolor", "normal", "height", "roughness", "metallic"]:
            map_value = getattr(self, attr)
            maps[attr] = map_value.cpu().numpy() if map_value is not None else None
        return maps

    def to_pil(self):
        """
        Convert all texture maps to PIL Images.

        Returns:
            dict: A dictionary containing PIL Images of the texture maps.
        """
        maps = {}
        for attr in ["basecolor", "normal", "height", "roughness", "metallic"]:
            map_value = getattr(self, attr)
            if map_value is not None:
                if attr == "normal":
                    # Scale the normal map from [-1, 1] to [0, 1] before converting to PIL
                    map_value = (map_value + 1.0) * 0.5
                maps[attr] = TF.to_pil_image(map_value.cpu())
            else:
                maps[attr] = None
        return maps

    def _to_tensor(
        self, image: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]]
    ) -> Optional[torch.FloatTensor]:
        """
        Convert an image to a torch tensor.

        Args:
            image: The image to convert.

        Returns:
            torch.FloatTensor: The image as a tensor.
        """
        if image is None:
            return None
        if isinstance(image, torch.FloatTensor):
            return image.cpu()
        elif isinstance(image, np.ndarray):
            return torch.from_numpy(image).float()
        elif isinstance(image, Image.Image):
            return TF.to_tensor(image)
        else:
            raise TypeError(f"Unsupported type: {type(image)}")

    def __repr__(self):
        return (
            f"Material(basecolor={self.basecolor.shape if self.basecolor is not None else None}, "
            f"normal={self.normal.shape if self.normal is not None else None}, "
            f"height={self.height.shape if self.height is not None else None}, "
            f"roughness={self.roughness.shape if self.roughness is not None else None}, "
            f"metallic={self.metallic.shape if self.metallic is not None else None})"
        )

    def save_to_folder(self, folder_path: str):
        """
        Save the material maps to a folder.

        Args:
            folder_path: The path to the folder where maps will be saved.
        """
        from .io import save_material_to_folder

        save_material_to_folder(self, folder_path)
