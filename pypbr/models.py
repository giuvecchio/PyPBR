"""
pypbr.models

This module defines various Bidirectional Reflectance Distribution Function (BRDF) models
used in Physically Based Rendering (PBR). It includes abstract base classes and concrete
implementations like the Cook-Torrance model, facilitating material rendering.

Classes:
    BRDFModel: Abstract base class for BRDF models.
    CookTorranceBRDF: Implementation of the Cook-Torrance BRDF model.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .material import MaterialBase
from .utils import linear_to_srgb


class BRDFModel(nn.Module, ABC):
    """
    Abstract base class for BRDF models.
    """


class CookTorranceBRDF(BRDFModel):
    """
    Implements the Cook-Torrance BRDF model.
    Supports both directional and point light sources.

    Example:
        ```python
        brdf = CookTorranceBRDF(light_type='directional')
        color = brdf(material, view_dir, light_dir, light_intensity)
        ```
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
        # Determine the device from the material
        device = self.override_device or material.device

        # Move tensors to the correct device
        view_dir = view_dir.to(device)
        light_intensity = light_intensity.to(device)

        # Normalize view direction
        view_dir = F.normalize(view_dir, dim=0)

        # Get the material properties
        # Common properties
        roughness = material.roughness.to(device)  # Shape (1, H, W)
        normal_map = material.normal.to(device) if material.normal is not None else None

        # Determine the workflow based on material properties
        if hasattr(material, "metallic") and material.metallic is not None:
            # Basecolor-Metallic workflow
            basecolor = material.linear_albedo.to(device)  # Shape (3, H, W)
            metallic = material.metallic.to(device)  # Shape (1, H, W)

            # F0 is interpolated between dielectric and conductor
            F0 = torch.lerp(torch.full_like(basecolor, 0.04), basecolor, metallic)
        elif hasattr(material, "specular") and material.specular is not None:
            # Diffuse-Specular workflow
            basecolor = material.linear_albedo.to(
                device
            )  # Diffuse color, shape (3, H, W)
            specular = material.linear_specular.to(device)  # Shape (3, H, W)

            # F0 is the specular map
            F0 = specular
            metallic = None  # Metallic is not used in this workflow
        else:
            raise ValueError(
                "Material must have either 'metallic' or 'specular' property."
            )

        # Get the size of the texture maps
        _, H, W = basecolor.shape

        # Expand view direction to match the spatial dimensions
        view_dir_map = view_dir.view(3, 1, 1).expand(3, H, W)

        # Initialize attenuation
        attenuation = 1.0

        # Expand light intensity to match spatial dimensions
        light_intensity = light_intensity.view(3, 1, 1)

        if self.light_type == "directional":
            # For directional light, light_dir_or_position is light_dir
            light_dir = light_dir_or_position.to(device)
            light_dir = F.normalize(light_dir, dim=0)
            light_dir_map = light_dir.view(3, 1, 1).expand(3, H, W)
        elif self.light_type == "point":
            # For point light, light_dir_or_position is light_position
            light_position = light_dir_or_position.to(device).view(3, 1, 1)

            # Generate positions grid for the surface points
            light_size = light_size or 1.0
            x = torch.linspace(-light_size / 2, light_size / 2, W, device=device)
            y = torch.linspace(-light_size / 2, light_size / 2, H, device=device)

            # Surface positions
            yv, xv = torch.meshgrid(y, x, indexing="ij")  # Correct 'ij' indexing
            zv = torch.zeros_like(xv)
            positions = torch.stack([xv, -yv, zv], dim=0)  # Shape (3, H, W)

            # Compute light direction and distance
            light_dir_map = light_position - positions  # Shape (3, H, W)
            distances = torch.norm(
                light_dir_map, dim=0, keepdim=True
            )  # Shape (1, H, W)
            light_dir_map = light_dir_map / (
                distances + 1e-7
            )  # Normalize to get direction

            # Compute attenuation (inverse square law)
            attenuation = 1.0 / (distances**2 + 1e-7)  # Shape (1, H, W)
        else:
            raise ValueError("Invalid light_type")

        # Use the normal map if provided, else use default normals
        if normal_map is not None:
            normal = normal_map.to(device)
        else:
            # Default normals pointing in +Z direction
            normal = (
                torch.tensor([0.0, 0.0, 1.0], device=device)
                .view(3, 1, 1)
                .expand(3, H, W)
            )

        # Normalize normals
        normal = F.normalize(normal, dim=0)

        # Compute half vector
        half_vector = F.normalize(
            view_dir_map + light_dir_map, dim=0
        )  # Shape (3, H, W)

        # Fresnel term
        cos_theta = torch.clamp(
            (half_vector * view_dir_map).sum(dim=0, keepdim=True), 0.0, 1.0
        )
        Fs = self.fresnel_schlick(cos_theta, F0)

        # Normal Distribution Function (NDF)
        NDF = self.normal_distribution_ggx(normal, half_vector, roughness)

        # Geometry Function
        G = self.geometry_smith(normal, view_dir_map, light_dir_map, roughness)

        # Denominator
        NdotV = torch.clamp((normal * view_dir_map).sum(dim=0, keepdim=True), 0.0, 1.0)
        NdotL = torch.clamp((normal * light_dir_map).sum(dim=0, keepdim=True), 0.0, 1.0)
        denom = 4.0 * NdotV * NdotL + 1e-7

        # Specular term
        specular = (Fs * NDF * G) / denom  # Shape (3, H, W)

        # Compute kD (diffuse component) differently based on the workflow
        if hasattr(material, "metallic") and material.metallic is not None:
            # Metallic-Roughness workflow
            kD = (1.0 - Fs) * (1.0 - metallic)  # Shape (3, H, W)
        else:
            # Diffuse-Specular workflow
            kD = 1.0 - Fs  # Shape (3, H, W)

        # Lambertian diffuse
        diffuse = kD * basecolor / torch.pi  # Shape (3, H, W)

        # Final BRDF
        # Ensure that light_intensity, NdotL, and attenuation are broadcastable
        radiance = light_intensity * (NdotL * attenuation)  # Shape (3, H, W)

        # Final color
        color = (diffuse + specular) * radiance  # Shape (3, H, W)

        # Ensure the color is in [0,1]
        color = torch.clamp(color, 0.0, 1.0)

        # Convert to sRGB if needed
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

        The GGX NDF is used to model the distribution of microfacets on a surface.

        Args:
            normal (Tensor): Surface normals (N), shape (3, H, W).
            half_vector (Tensor): Half vectors (H), shape (3, H, W).
            roughness (Tensor): Surface roughness, shape (1, H, W).

        Returns:
            Tensor: NDF term, shape (1, H, W).

        References:
            Walter, B., Marschner, S.R., Li, H., and Kautz, J. (2007). Microfacet Models for Refraction through Rough Surfaces.
            Journal of Computer Graphics Techniques (JCGT).
        """
        a = roughness**2
        NdotH = torch.clamp((normal * half_vector).sum(dim=0, keepdim=True), 0.0, 1.0)
        a2 = a**2
        denom = NdotH**2 * (a2 - 1.0) + 1.0
        NDF = a2 / (torch.pi * denom**2 + 1e-7)
        return NDF

    def geometry_schlick_ggx(self, NdotX: Tensor, roughness: Tensor) -> Tensor:
        """
        Compute the geometry function for a single direction using Schlick-GGX.

        Args:
            NdotX (Tensor): Cosine of angle between normal and direction, shape (1, H, W).
            roughness (Tensor): Surface roughness, shape (1, H, W).

        Returns:
            Tensor: Geometry term for one direction, shape (1, H, W).
        """
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
