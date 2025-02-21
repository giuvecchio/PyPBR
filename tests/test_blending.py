import torch

from pypbr.blending.functional import blend_with_mask
from pypbr.materials import MaterialBase


def create_dummy_material(color_value, H=32, W=32):
    # Create a dummy material with constant albedo and random normals
    material = MaterialBase(
        albedo=torch.full((3, H, W), color_value),
        normal=torch.rand(3, H, W) * 2 - 1,
        roughness=torch.rand(1, H, W),
    )
    return material


def test_blend_with_mask():
    H, W = 32, 32
    material1 = create_dummy_material(0.2, H, W)
    material2 = create_dummy_material(0.8, H, W)
    mask = torch.ones(1, H, W) * 0.5
    blended, _ = blend_with_mask(material1, material2, mask)
    # For a 50-50 blend, the albedo should be roughly the average
    blended_albedo = blended.albedo
    expected = (0.2 + 0.8) / 2
    assert torch.allclose(blended_albedo.mean(), torch.tensor(expected), atol=0.1)
