"""
pypbr.models.cooktorrance

This module defines various Bidirectional Reflectance Distribution Function (BRDF) models
used in Physically Based Rendering (PBR). It includes abstract base classes and concrete
implementations like the Cook-Torrance model, facilitating material rendering.

Classes:
    BRDFModel: Abstract base class for BRDF models.
    
    CookTorranceBRDF: Implementation of the Cook-Torrance BRDF model.
"""

from abc import ABC, abstractmethod
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..materials import MaterialBase
from ..utils import linear_to_srgb


class BRDFModel(nn.Module, ABC):
    """
    Abstract base class for BRDF models.
    """


class CookTorranceBRDF(BRDFModel):
    """
    Implements the Cook-Torrance BRDF model.
    Supports both directional and point light sources.

    Example:
        .. code-block:: python

            brdf = CookTorranceBRDF(light_type="point")

            # Define the view direction, light direction, and light intensity
            view_dir = torch.tensor([0.0, 0.0, 1.0])  # Viewing straight on
            light_dir = torch.tensor([0.1, 0.1, 1.0])  # Light coming from slightly top right
            light_intensity = torch.tensor([1.0, 1.0, 1.0])  # White light
            light_size = 1.0

            # Evaluate the BRDF to get the reflected color
            color = brdf(material, view_dir, light_dir, light_intensity, light_size)
    """

    def __init__(self, light_type: str = "point", override_device: torch.device = None):
        """
        Initialize the Cook-Torrance BRDF.

        Args:
            light_type (str): Type of light source ('directional' or 'point').
        """
        super().__init__()
        self.light_type = light_type.lower()
        if self.light_type not in ["directional", "point"]:
            raise ValueError(
                f"Unsupported light_type: {self.light_type}. Must be 'directional' or 'point'."
            )
        self.override_device = override_device

    def forward(
        self,
        material: MaterialBase,
        view_dir: Tensor,
        light_dir_or_position: Tensor,
        light_intensity: Tensor,
        light_size: Optional[float] = None,
        return_srgb: bool = True,
    ) -> Tensor:
        """
        Evaluate the Cook-Torrance BRDF for the given directions.

        Args:
            material (MaterialBase): Material properties (can be BasecolorMetallicMaterial or DiffuseSpecularMaterial).
            view_dir (Tensor): View direction vector, shape (3,).
            light_dir_or_position (Tensor): Light direction vector (for directional light, shape (3,))
                                            or light position vector (for point light, shape (3,)).
            light_intensity (Tensor): Light intensity, shape (3,).
            light_size (float): Size of the light source (for point light only).
            return_srgb (bool): Whether to return the color in sRGB space.

        Returns:
            Tensor: The reflected color at each point, shape (3, H, W).
        """
        # Determine the device
        device = self.override_device or material.device

        view_dir = F.normalize(view_dir.to(device), dim=0)
        light_intensity = light_intensity.to(device).view(3, 1, 1)

        # Get material properties
        roughness = material.roughness.to(device)  # Shape (1, H, W)
        normal_map = material.normal.to(device) if material.normal is not None else None

        # Determine workflow: metallic or specular
        if hasattr(material, "metallic") and material.metallic is not None:
            basecolor = material.linear_albedo.to(device)  # Shape (3, H, W)
            metallic = material.metallic.to(device)  # Shape (1, H, W)
            # Interpolate F0 between dielectric (0.04) and basecolor
            F0 = torch.lerp(torch.full_like(basecolor, 0.04), basecolor, metallic)
        elif hasattr(material, "specular") and material.specular is not None:
            basecolor = material.linear_albedo.to(
                device
            )  # Diffuse color, shape (3, H, W)
            specular = material.linear_specular.to(device)  # Shape (3, H, W)
            F0 = specular
            metallic = None
        else:
            raise ValueError(
                "Material must have either 'metallic' or 'specular' property."
            )

        _, H, W = basecolor.shape
        view_dir_map = view_dir.view(3, 1, 1).expand(3, H, W)
        attenuation = 1.0  # Default attenuation for directional light

        # Handle light source type
        if self.light_type == "directional":
            light_dir = F.normalize(light_dir_or_position.to(device), dim=0)
            light_dir_map = light_dir.view(3, 1, 1).expand(3, H, W)
        elif self.light_type == "point":
            light_position = light_dir_or_position.to(device).view(3, 1, 1)
            light_size = light_size or 1.0
            # Create a grid of surface positions
            x = torch.linspace(-light_size / 2, light_size / 2, W, device=device)
            y = torch.linspace(-light_size / 2, light_size / 2, H, device=device)
            yv, xv = torch.meshgrid(y, x, indexing="ij")
            zv = torch.zeros_like(xv)
            positions = torch.stack([xv, -yv, zv], dim=0)  # Shape (3, H, W)
            light_dir_map = light_position - positions
            distances = torch.norm(light_dir_map, dim=0, keepdim=True)
            light_dir_map = light_dir_map / (distances + 1e-7)
            attenuation = 1.0 / (distances**2 + 1e-7)
        else:
            raise ValueError("Invalid light_type")

        # Use provided normal map or default to a +Z normal
        if normal_map is not None:
            normal = normal_map.to(device)
        else:
            normal = (
                torch.tensor([0.0, 0.0, 1.0], device=device)
                .view(3, 1, 1)
                .expand(3, H, W)
            )

        normal = F.normalize(normal, dim=0)
        half_vector = F.normalize(view_dir_map + light_dir_map, dim=0)
        cos_theta = torch.clamp(
            (half_vector * view_dir_map).sum(dim=0, keepdim=True), 0.0, 1.0
        )
        Fs = self.fresnel_schlick(cos_theta, F0)
        NDF = self.normal_distribution_ggx(normal, half_vector, roughness)
        G = self.geometry_smith(normal, view_dir_map, light_dir_map, roughness)

        NdotV = torch.clamp((normal * view_dir_map).sum(dim=0, keepdim=True), 0.0, 1.0)
        NdotL = torch.clamp((normal * light_dir_map).sum(dim=0, keepdim=True), 0.0, 1.0)
        denom = 4.0 * NdotV * NdotL + 1e-7
        specular = (Fs * NDF * G) / denom

        # Compute diffuse component based on workflow
        if metallic is not None:
            kD = (1.0 - Fs) * (1.0 - metallic)
        else:
            kD = 1.0 - Fs

        diffuse = kD * basecolor / math.pi
        radiance = light_intensity * (NdotL * attenuation)
        color = (diffuse + specular) * radiance
        color = torch.clamp(color, 0.0, 1.0)

        if return_srgb:
            color = linear_to_srgb(color)

        return color

    def fresnel_schlick(self, cos_theta: Tensor, F0: Tensor) -> Tensor:
        """
        Compute the Fresnel term using Schlick's approximation.

        Args:
            cos_theta (Tensor): Cosine of the angle between view and half-vector, shape (1, H, W).
            F0 (Tensor): Base reflectivity at normal incidence, shape (3, H, W).

        Returns:
            Tensor: Fresnel term, shape (3, H, W).
        """
        cos_theta = cos_theta.expand_as(F0)
        return F0 + (1.0 - F0) * torch.pow(1.0 - cos_theta, 5.0)

    def normal_distribution_ggx(
        self, normal: Tensor, half_vector: Tensor, roughness: Tensor
    ) -> Tensor:
        """
        Compute the Normal Distribution Function using the GGX (Trowbridge-Reitz) model.

        Args:
            normal (Tensor): Surface normals (N), shape (3, H, W).
            half_vector (Tensor): Half vectors (H), shape (3, H, W).
            roughness (Tensor): Surface roughness, shape (1, H, W).

        Returns:
            Tensor: NDF term, shape (1, H, W).
        """
        # Use roughness as the microfacet parameter alpha
        alpha = roughness
        alpha2 = alpha * alpha
        NdotH = torch.clamp((normal * half_vector).sum(dim=0, keepdim=True), 0.0, 1.0)
        denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0
        NDF = alpha2 / (torch.pi * (denom**2) + 1e-7)
        return NDF

    def geometry_schlick_ggx(self, NdotX: Tensor, roughness: Tensor) -> Tensor:
        """
        Compute the geometry function for a single direction using Schlick-GGX.

        Args:
            NdotX (Tensor): Cosine of the angle between normal and direction, shape (1, H, W).
            roughness (Tensor): Surface roughness, shape (1, H, W).

        Returns:
            Tensor: Geometry term for one direction, shape (1, H, W).
        """
        # Using the common Schlick-GGX approximation.
        r = roughness + 1.0
        k = (r**2) / 8.0
        denom = NdotX * (1.0 - k) + k + 1e-7
        return NdotX / denom

    def geometry_smith(
        self,
        normal: Tensor,
        view_dir_map: Tensor,
        light_dir_map: Tensor,
        roughness: Tensor,
    ) -> Tensor:
        """
        Compute the geometry function using Smith's method.

        Args:
            normal (Tensor): Surface normals (N), shape (3, H, W).
            view_dir_map (Tensor): View directions (V), shape (3, H, W).
            light_dir_map (Tensor): Light directions (L), shape (3, H, W).
            roughness (Tensor): Surface roughness, shape (1, H, W).

        Returns:
            Tensor: Geometry term, shape (1, H, W).
        """
        NdotV = torch.clamp((normal * view_dir_map).sum(dim=0, keepdim=True), 0.0, 1.0)
        NdotL = torch.clamp((normal * light_dir_map).sum(dim=0, keepdim=True), 0.0, 1.0)
        ggx1 = self.geometry_schlick_ggx(NdotV, roughness)
        ggx2 = self.geometry_schlick_ggx(NdotL, roughness)
        return ggx1 * ggx2
