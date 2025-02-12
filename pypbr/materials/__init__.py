"""
pypbr.materials

This module provides a set of classes and functions for creating and manipulating Physically Based Rendering (PBR) materials. 

Classes:
    `MaterialBase`: Base class representing a PBR material.
    `BasecolorMetallicMaterial`: Represents a PBR material using basecolor and metallic maps.
    `DiffuseSpecularMaterial`: Represents a PBR material using diffuse and specular maps.
"""

from .base import MaterialBase
from .metallic import BasecolorMetallicMaterial
from .diffuse import DiffuseSpecularMaterial

__all__ = [
    "MaterialBase",
    "BasecolorMetallicMaterial",
    "DiffuseSpecularMaterial",
]
