import os

import pytest
import torch

from pypbr.blending.functional import blend_with_mask
from pypbr.io import load_material_from_folder


@pytest.fixture
def rocks_material_dir():
    import os

    return os.path.join(os.path.dirname(__file__), "data", "rocks")


@pytest.fixture
def tiles_material_dir():
    import os

    return os.path.join(os.path.dirname(__file__), "data", "tiles")


def test_blend_with_mask(rocks_material_dir, tiles_material_dir):
    # Load two real materials from the dataset.
    mat1 = load_material_from_folder(rocks_material_dir)
    mat2 = load_material_from_folder(tiles_material_dir)

    # Create a 50-50 blending mask.
    H, W = mat1.size
    mask = torch.full((1, H, W), 0.5)

    # Blend the two materials.
    blended, _ = blend_with_mask(mat1, mat2, mask)

    # For a 50-50 blend, the albedo should be roughly the average of the two.
    # (Here we compare the mean intensity of the albedo.)
    avg1 = mat1.albedo.mean().item()
    avg2 = mat2.albedo.mean().item()
    expected = (avg1 + avg2) / 2
    blended_avg = blended.albedo.mean().item()
    assert abs(blended_avg - expected) < 0.1
