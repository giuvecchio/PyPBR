"""
pypbr.utils.enums

This module provides utility enums for Physically Based Rendering (PBR) materials.

Classes:
    `NormalConvention`: Defines the normal convention for a given material.
"""

from enum import Enum

class NormalConvention(Enum):
    OPENGL = 0   # Upwards facing normals (e.g. [0,0,1])
    DIRECTX = 1  # Typically, inverted Y axis