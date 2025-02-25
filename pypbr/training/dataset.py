"""
pypbr.training.dataset

Provides a PyTorch Dataset class for loading PBR materials.
Each subdirectory in the root directory is assumed to contain a single material, which is loaded using pypbr.io.
"""

import os

import torch
from torch.utils.data import Dataset

from ..io import load_material_from_folder


class PBRMaterialDataset(Dataset):
    """
    A PyTorch Dataset class for loading PBR materials using the PyPBR library.

    Each subdirectory in the root directory is assumed to be a material folder.
    """

    def __init__(self, root_dir, preferred_workflow=None, transform=None, is_srgb=True):
        """
        Args:
            root_dir (str): Root directory containing material folders.
            preferred_workflow (str, optional): 'metallic' or 'specular'. Determines which maps to use.
            transform (callable, optional): An optional transform to apply to each material.
            is_srgb (bool): Whether the albedo (and specular) maps are in sRGB space.
        """
        self.root_dir = root_dir
        self.material_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.preferred_workflow = preferred_workflow
        self.transform = transform
        self.is_srgb = is_srgb

    def __len__(self):
        return len(self.material_dirs)

    def __getitem__(self, idx):
        material_folder = self.material_dirs[idx]
        # Use your load_material_from_folder function to load the material.
        material = load_material_from_folder(
            material_folder,
            preferred_workflow=self.preferred_workflow,
            is_srgb=self.is_srgb,
        )
        if self.transform:
            material = self.transform(material)
        return material
