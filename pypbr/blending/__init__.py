"""
pypbr.blending

This module provides a set of classes and functions for blending Physically Based Rendering (PBR) materials in different ways. The blending can be done using masks, height maps, properties, or gradients, and can be applied through both class-based and functional approaches.

Contents:
    - **Classes**: Object-oriented blending methods.
    - **Functional API**: Function-based blending utilities for quick use.
"""

from . import functional
from .blending import (
    BlendFactory,
    BlendMethod,
    GradientBlend,
    HeightBlend,
    MaskBlend,
    PropertyBlend,
)

__all__ = [
    "functional",
    "BlendMethod",
    "MaskBlend",
    "HeightBlend",
    "PropertyBlend",
    "GradientBlend",
    "BlendFactory",
]
