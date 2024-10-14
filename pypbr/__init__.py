"""
pypbr.transforms

This module provides transformations for PBR materials, similar to torchvision's transforms.
It includes both functional APIs and transformation classes that can be composed.

Modules:
    blending: Classes for blending materials using various methodologies.
    io: Input/output functions for saving and loading materials.
    material: Classes representing PBR materials.
    models: Classes representing BRDF models.
    utils: Utility functions for color space conversion.

Classes:
    MaterialBase,
    BasecolorMetallicMaterial,
    DiffuseSpecularMaterial,
    BRDFModel,
    CookTorranceBRDF,

Functions:
    load_material_from_folder,
    save_material_to_folder,
    linear_to_srgb,
    srgb_to_linear,
"""

from . import blending, io, utils
from ._version import version as __version__
from .material import BasecolorMetallicMaterial, DiffuseSpecularMaterial, MaterialBase
from .models import BRDFModel, CookTorranceBRDF

__all__ = [
    "blending",
    "io",
    "utils",
    "MaterialBase",
    "BasecolorMetallicMaterial",
    "DiffuseSpecularMaterial",
    "BRDFModel",
    "CookTorranceBRDF",
]
