"""
pypbr.utils

This module provides a set of classes and functions for creating and manipulating Physically Based Rendering (PBR) materials. 

Classes:
    `MaterialBase`: Base class representing a PBR material.
    `BasecolorMetallicMaterial`: Represents a PBR material using basecolor and metallic maps.
    `DiffuseSpecularMaterial`: Represents a PBR material using diffuse and specular maps.
"""

from .enums import *
from .functions import *

__all__ = [
    "NormalConvention",
    "srgb_to_linear",
    "linear_to_srgb",
    "rotate_normals",
    "invert_normal",
    "compute_normal_from_height",
    "compute_height_from_normal",
]
