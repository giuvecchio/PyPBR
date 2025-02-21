"""
pypbr.materials.metallic

This module defines the `BasecolorMetallicMaterial`. It provides functionalities to manipulate
and convert these texture maps for rendering purposes.

Classes:
    `BasecolorMetallicMaterial`: Represents a PBR material using basecolor and metallic maps.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from .base import MaterialBase


class BasecolorMetallicMaterial(MaterialBase):
    """
    A class representing a PBR material using basecolor and metallic maps.

    Attributes:
        albedo (torch.FloatTensor): The albedo map tensor.
        normal (torch.FloatTensor): The normal map tensor.
        roughness (torch.FloatTensor): The roughness map tensor.
        metallic (torch.FloatTensor): The metallic map tensor.
    """

    def __init__(
        self,
        albedo: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        albedo_is_srgb: bool = True,
        normal: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        roughness: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        metallic: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        **kwargs,
    ):
        """
        Initialize the Material with optional texture maps.

        Args:
            albedo: The albedo map.
            albedo_is_srgb: Flag indicating if albedo is in sRGB space.
            normal: The normal map.
            roughness: The roughness map.
            metallic: The metallic map.
            **kwargs: Additional texture maps.
        """
        super().__init__(
            albedo=albedo,
            albedo_is_srgb=albedo_is_srgb,
            normal=normal,
            roughness=roughness,
            **kwargs,
        )
        if metallic is not None:
            self.metallic = metallic

    @property
    def basecolor(self):
        return self.albedo

    @basecolor.setter
    def basecolor(self, value):
        self.albedo = value

    def to_diffuse_specular_material(self, albedo_is_srgb: bool = False):
        """
        Convert the material from basecolor-metallic workflow to diffuse-specular workflow.

        Args:
            albedo_is_srgb: Flag indicating if the albedo map should be returned in sRGB space.

        Returns:
            DiffuseSpecularMaterial: A new material instance in the diffuse-specular workflow.
        """
        from .diffuse import DiffuseSpecularMaterial

        # Ensure albedo and metallic maps are available
        if self.albedo is None or self.metallic is None:
            raise ValueError(
                "Both albedo and metallic maps are required for conversion."
            )

        # Convert albedo to linear space if necessary
        albedo = self.linear_albedo

        # Resize metallic map to match albedo size if necessary
        if self.metallic.shape[1:] != albedo.shape[1:]:
            metallic = TF.resize(self.metallic, albedo.shape[1:])
        else:
            metallic = self.metallic

        # Ensure metallic map has the correct dimensions
        if metallic.dim() == 2:
            metallic = metallic.unsqueeze(0)  # Add channel dimension

        # Specular dielectric color (F0 for non-metals)
        specular_dielectric = torch.full_like(albedo, 0.04)  # 4% reflectance

        # Compute diffuse color
        diffuse = albedo * (1.0 - metallic)

        # Compute specular color
        specular = specular_dielectric * (1.0 - metallic) + albedo * metallic

        # Create a new DiffuseSpecularMaterial instance
        diffuse_specular_material = DiffuseSpecularMaterial(
            albedo=diffuse,
            specular=specular,
            normal=self.normal,
            roughness=self.roughness,
            albedo_is_srgb=albedo_is_srgb,  # The diffuse map is in linear space
        )

        return diffuse_specular_material
