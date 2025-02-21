import os
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from pypbr.io import load_material_from_folder, save_material_to_folder
from pypbr.materials import BasecolorMetallicMaterial


@pytest.fixture
def temp_material_dir(tmp_path):
    # Create a temporary directory and save dummy images.
    d = tmp_path / "material"
    d.mkdir()
    # Create a dummy basecolor image
    basecolor = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    Image.fromarray(basecolor).save(d / "albedo.png")
    # Create a dummy normal image
    normal = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    Image.fromarray(normal).save(d / "normal.png")
    # Create a dummy roughness image (grayscale)
    roughness = (np.random.rand(64, 64) * 255).astype(np.uint8)
    Image.fromarray(roughness).save(d / "roughness.png")
    return d


def test_io_load_and_save(temp_material_dir, tmp_path):
    material = load_material_from_folder(str(temp_material_dir))
    # Modify material (e.g., convert to linear)
    material.to_linear()
    # Save to another temporary directory
    out_dir = tmp_path / "output_material"
    out_dir.mkdir()
    material.save_to_folder(str(out_dir))
    # Check that expected files are saved
    for fname in ["albedo.png", "normal.png", "roughness.png"]:
        assert (out_dir / fname).is_file()
