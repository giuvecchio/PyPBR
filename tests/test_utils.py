import numpy as np
import torch

from pypbr.utils import linear_to_srgb, srgb_to_linear


def test_linear_srgb_roundtrip():
    # Create a dummy texture in sRGB space
    texture = torch.linspace(0, 1, steps=100).view(1, 10, 10)
    linear = srgb_to_linear(texture)
    srgb = linear_to_srgb(linear)
    # Check that round-trip conversion stays close
    assert torch.allclose(texture, srgb, atol=1e-4)
