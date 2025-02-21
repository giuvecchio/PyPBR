"""
pypbr.materials.diffuse

This module defines the `DiffuseSpecularMaterial`. It provides functionalities to manipulate
and convert these texture maps for rendering purposes.

Classes:
    `DiffuseSpecularMaterial`: Represents a PBR material using diffuse and specular maps.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from ..utils import linear_to_srgb, srgb_to_linear
from .base import MaterialBase


class DiffuseSpecularMaterial(MaterialBase):
    """
    A class representing a PBR material using diffuse and specular maps.

    Attributes:
        albedo (torch.FloatTensor): The albedo map tensor.
        normal (torch.FloatTensor): The normal map tensor.
        roughness (torch.FloatTensor): The roughness map tensor.
        specular (torch.FloatTensor): The specular map tensor.
    """

    def __init__(
        self,
        albedo: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        albedo_is_srgb: bool = True,
        normal: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        roughness: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        specular: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        specular_is_srgb: bool = True,
        **kwargs,
    ):
        """
        Initialize the Material with optional texture maps.

        Args:
            albedo: The albedo map.
            albedo_is_srgb: Flag indicating if albedo is in sRGB space.
            normal: The normal map.
            roughness: The roughness map.
            specular: The specular map.
            specular_is_srgb: Flag indicating if specular is in sRGB space.
            **kwargs: Additional texture maps.
        """
        super().__init__(
            albedo=albedo,
            albedo_is_srgb=albedo_is_srgb,
            normal=normal,
            roughness=roughness,
            **kwargs,
        )

        self.specular_is_srgb = specular_is_srgb
        if specular is not None:
            self.specular = specular

    @property
    def diffuse(self):
        return self.albedo

    @diffuse.setter
    def diffuse(self, value):
        self.albedo = value

    @property
    def linear_specular(self):
        """
        Get the specular map in linear space.

        Returns:
            torch.FloatTensor: The albedo map in linear space.
        """
        specular = self._maps.get("specular", None)
        if specular is not None:
            if self.specular_is_srgb:
                return srgb_to_linear(specular)
            else:
                return specular
        else:
            return None

    def to_basecolor_metallic_material(self, albedo_is_srgb: bool = False):
        """
        Convert the material from diffuse-specular workflow to basecolor-metallic workflow.

        Args:
            albedo_is_srgb: Flag indicating if the albedo map should be returned in sRGB space.

        Returns:
            BasecolorMetallicMaterial: A new material instance in the basecolor-metallic workflow.
        """
        from .metallic import BasecolorMetallicMaterial

        # Ensure diffuse and specular maps are available
        if self.albedo is None or self.specular is None:
            raise ValueError(
                "Both albedo (diffuse) and specular maps are required for conversion."
            )

        # Convert albedo (diffuse) to linear space if necessary
        diffuse = self.linear_albedo

        # Specular dielectric color (F0 for non-metals)
        specular_dielectric = 0.04  # 4% reflectance

        # Resize specular map to match diffuse size if necessary
        if self.specular.shape[1:] != diffuse.shape[1:]:
            specular = TF.resize(self.specular, diffuse.shape[1:], antialias=True)
        else:
            specular = self.specular

        # Ensure specular map has the correct dimensions
        if specular.dim() == 2:
            specular = specular.unsqueeze(0)  # Add channel dimension

        # Calculate metallic map
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-6
        numerator = specular - specular_dielectric
        denominator = diffuse - specular_dielectric + epsilon
        metallic = torch.clamp(numerator / (denominator + epsilon), 0.0, 1.0)

        # Handle edge cases where denominator is very small
        metallic = torch.where(
            denominator < epsilon, torch.zeros_like(metallic), metallic
        )

        # Compute basecolor
        basecolor = (diffuse) / (1.0 - metallic + epsilon)

        # Where metallic is close to 1, use specular as basecolor
        metallic_threshold = 0.95
        basecolor = torch.where(metallic >= metallic_threshold, specular, basecolor)

        # Clamp basecolor to valid range
        basecolor = torch.clamp(basecolor, 0.0, 1.0)

        # Create a new BasecolorMetallicMaterial instance
        basecolor_metallic_material = BasecolorMetallicMaterial(
            albedo=basecolor,
            metallic=metallic,
            normal=self.normal,
            roughness=self.roughness,
            albedo_is_srgb=albedo_is_srgb,  # The basecolor is in linear space
        )

        return basecolor_metallic_material

    def to_linear(self):
        """
        Convert the albedo and specular maps to linear space if it's in sRGB.

        Returns:
            MaterialBase: Returns self for method chaining.
        """

        super().to_linear()

        specular = self._maps.get("specular", None)
        if specular is not None and self.specular_is_srgb:
            self._maps["specular"] = srgb_to_linear(specular)
            self.specular_is_srgb = False

        return self

    def to_srgb(self):
        """
        Convert the albedo and specular maps to sRGB space if it's in linear space.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        super().to_srgb()
        specular = self._maps.get("specular", None)
        if specular is not None and not self.specular_is_srgb:
            self._maps["specular"] = linear_to_srgb(specular)
            self.specular_is_srgb = True

        return self
