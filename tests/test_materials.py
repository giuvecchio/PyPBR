import pytest
import torch

from pypbr.materials import (
    BasecolorMetallicMaterial,
    DiffuseSpecularMaterial,
    MaterialBase,
)


def create_dummy_material_metallic(H, W):
    albedo = torch.rand(3, H, W)
    normal = torch.rand(3, H, W) * 2 - 1  # values in [-1,1]
    roughness = torch.rand(1, H, W)
    metallic = torch.rand(1, H, W)
    return BasecolorMetallicMaterial(
        albedo=albedo, normal=normal, roughness=roughness, metallic=metallic
    )


def create_dummy_material_specular(H, W):
    albedo = torch.rand(3, H, W)
    normal = torch.rand(3, H, W) * 2 - 1
    roughness = torch.rand(1, H, W)
    specular = torch.rand(3, H, W)
    return DiffuseSpecularMaterial(
        albedo=albedo, normal=normal, roughness=roughness, specular=specular
    )


def test_as_tensor():
    H, W = 32, 32
    dummy_albedo = torch.rand(3, H, W)
    dummy_normal = torch.rand(3, H, W)
    dummy_roughness = torch.rand(1, H, W)
    material = MaterialBase(
        albedo=dummy_albedo, normal=dummy_normal, roughness=dummy_roughness
    )
    tensor = material.as_tensor(["albedo", ("normal", 3), ("roughness", 1)])
    # Expect total channels = 3 + 3 + 1 = 7
    assert tensor.shape[0] == 7
    assert tensor.shape[1:] == (H, W)


def test_clone_and_device():
    H, W = 16, 16
    material = create_dummy_material_metallic(H, W)
    clone = material.clone()
    # Check that each map is a different tensor instance
    for key in material._maps:
        assert material._maps[key] is not clone._maps[key]
    # Test device transfer (assuming CUDA is available, else use CPU)
    device = torch.device("cpu")
    clone.to(device)
    for key, map_value in clone._maps.items():
        if map_value is not None:
            assert map_value.device == device
