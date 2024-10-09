"""
pypbr.blending.blending

This module provides class-based interfaces for blending PBR materials using various methodologies.

Classes:
    BlendMethod: Abstract base class for different blending methods.
    MaskBlend: Blend two materials using a provided mask.
    HeightBlend: Blend two materials based on their height maps.
    PropertyBlend: Blend two materials based on a specified property map.
    GradientBlend: Blend two materials using a linear gradient mask.
    BlendFactory: Factory class to instantiate blending methods.
"""

from abc import ABC, abstractmethod

import torch

from ..material import MaterialBase
from .functional import (
    blend_on_height,
    blend_on_properties,
    blend_with_gradient,
    blend_with_mask,
)


class BlendMethod(ABC):
    """
    Abstract base class for different blending methods.
    """

    @abstractmethod
    def __call__(
        self, material1: MaterialBase, material2: MaterialBase
    ) -> MaterialBase:
        """
        Apply the blending method to two materials.

        Args:
            material1 (MaterialBase): The first material.
            material2 (MaterialBase): The second material.

        Returns:
            MaterialBase: A new material resulting from blending material1 and material2.
        """
        pass


class MaskBlend(BlendMethod):
    """
    Blend two materials using a provided mask.

    Args:
        mask (torch.FloatTensor): The blending mask tensor, with values in [0, 1], shape [1, H, W] or [H, W].

    Raises:
        ValueError: If the mask has an invalid shape.
    """

    def __init__(self, mask: torch.FloatTensor):
        if mask.dim() == 2:
            self.mask: torch.FloatTensor = mask.unsqueeze(0)
        elif mask.dim() == 3 and mask.size(0) == 1:
            self.mask = mask
        else:
            raise ValueError("Mask must have shape [1, H, W] or [H, W].")

    def __call__(
        self, material1: MaterialBase, material2: MaterialBase
    ) -> MaterialBase:
        """
        Apply mask-based blending to two materials.

        Args:
            material1 (MaterialBase): The first material.
            material2 (MaterialBase): The second material.

        Returns:
            MaterialBase: A new material resulting from blending material1 and material2.
        """
        return blend_with_mask(material1, material2, self.mask)


class HeightBlend(BlendMethod):
    """
    Blend two materials based on their height maps.

    Args:
        blend_width (float): Controls the sharpness of the blending transition.
        shift (float): Shifts the blending transition vertically.
                       Positive values raise the blending height,
                       while negative values lower it.
                       Defaults to 0.0.
    """

    def __init__(
        self,
        blend_width: float = 0.1,
        shift: float = 0.0,
    ):
        self.blend_width: float = blend_width
        self.shift: float = shift

    def __call__(
        self, material1: MaterialBase, material2: MaterialBase
    ) -> MaterialBase:
        """
        Apply height-based blending to two materials.

        Args:
            material1 (MaterialBase): The first material.
            material2 (MaterialBase): The second material.

        Returns:
            MaterialBase: A new material resulting from blending material1 and material2.
        """
        return blend_on_height(material1, material2, self.blend_width, self.shift)


class PropertyBlend(BlendMethod):
    """
    Blend two materials based on a specified property map.

    Args:
        property_name (str): The name of the property map to use for blending (e.g., 'metallic', 'roughness').
        blend_width (float): Controls the sharpness of the blending transition.
    """

    def __init__(self, property_name: str = "metallic", blend_width: float = 0.1):
        self.property_name: str = property_name
        self.blend_width: float = blend_width

    def __call__(
        self, material1: MaterialBase, material2: MaterialBase
    ) -> MaterialBase:
        """
        Apply property-based blending to two materials.

        Args:
            material1 (MaterialBase): The first material.
            material2 (MaterialBase): The second material.

        Returns:
            MaterialBase: A new material resulting from blending material1 and material2.
        """
        return blend_on_properties(
            material1, material2, self.property_name, self.blend_width
        )


class GradientBlend(BlendMethod):
    """
    Blend two materials using a linear gradient mask.

    Args:
        direction (str): The direction of the gradient ('horizontal' or 'vertical').

    Raises:
        ValueError: If an invalid direction is specified.
    """

    def __init__(self, direction: str = "horizontal"):
        if direction not in ["horizontal", "vertical"]:
            raise ValueError("Direction must be 'horizontal' or 'vertical'.")
        self.direction: str = direction

    def __call__(
        self, material1: MaterialBase, material2: MaterialBase
    ) -> MaterialBase:
        """
        Apply gradient-based blending to two materials.

        Args:
            material1 (MaterialBase): The first material.
            material2 (MaterialBase): The second material.

        Returns:
            MaterialBase: A new material resulting from blending material1 and material2.
        """
        return blend_with_gradient(material1, material2, self.direction)


class BlendFactory:
    """
    Factory class to instantiate blending methods.

    Methods:
        get_blend_method: Returns an instance of the specified blending method.
    """

    _BLEND_METHODS = {
        "mask": MaskBlend,
        "height": HeightBlend,
        "properties": PropertyBlend,
        "gradient": GradientBlend,
    }

    @staticmethod
    def get_blend_method(method_name: str, **kwargs) -> BlendMethod:
        """
        Factory method to get a blending method instance.

        Args:
            method_name (str): Name of the blending method ('mask', 'height', 'properties', 'gradient').
            **kwargs: Parameters for the blending method.

        Returns:
            BlendMethod: An instance of a BlendMethod subclass.

        Raises:
            ValueError: If an unknown blending method is specified.
        """
        method_class = BlendFactory._BLEND_METHODS.get(method_name.lower())
        if not method_class:
            raise ValueError(f"Unknown blending method: {method_name}")
        return method_class(**kwargs)
