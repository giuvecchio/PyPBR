import os
import shutil

import pytest
import torch
from PIL import Image

from pypbr.io import load_material_from_folder, save_material_to_folder
from pypbr.materials import BasecolorMetallicMaterial


@pytest.fixture
def rocks_material_dir():
    import os

    return os.path.join(os.path.dirname(__file__), "data", "rocks")


def test_io_load_and_save(rocks_material_dir, tmp_path):
    # Load material from the real "rocks" folder.
    material = load_material_from_folder(rocks_material_dir)
    # Convert to linear space, etc.
    material.to_linear()
    # Save to a temporary directory.
    out_dir = tmp_path / "output_material"
    out_dir.mkdir()
    material.save_to_folder(str(out_dir))
    # Check that expected files are saved.
    for fname in ["albedo.png", "normal.png", "roughness.png"]:
        assert (out_dir / fname).is_file()
