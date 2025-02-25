"""
pypbr: A Physically Based Rendering (PBR) Library

pypbr provides a comprehensive suite of tools for working with PBR materials, including:
  - Material definitions and workflows (basecolor‑metallic, diffuse‑specular)
  - Blending methods for combining material maps
  - I/O utilities for loading and saving materials
  - BRDF models for realistic shading
  - Transformation functions for material preprocessing
  - Utility functions for color space conversions and normal map operations

Modules:
  - **blending**: Blending operations for PBR materials.
  - **io**: Material input/output functionalities.
  - **materials**: Core material classes and workflows.
  - **models**: Implementation of BRDF models.
  - **transforms**: Geometric and photometric transformations for materials.
  - **utils**: Utility functions and enumerations.
"""

from . import blending, io, materials, models, transforms, utils
from ._version import version as __version__

__all__ = [
    "blending",
    "io",
    "materials",
    "models",
    "transforms",
    "utils",
]
