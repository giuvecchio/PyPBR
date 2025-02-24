import pytest
import torch

from pypbr.materials import (
    BasecolorMetallicMaterial,
    DiffuseSpecularMaterial,
    MaterialBase,
)
from pypbr.models import CookTorranceBRDF, LambertianBRDF, OrenNayarBRDF


def test_cooktorrance_directional():
    H, W = 64, 64
    material = BasecolorMetallicMaterial(
        albedo=torch.rand(3, H, W),
        normal=torch.rand(3, H, W) * 2 - 1,
        roughness=torch.rand(1, H, W),
        metallic=torch.rand(1, H, W),
    )
    brdf = CookTorranceBRDF(light_type="directional")
    view_dir = torch.tensor([0.0, 0.0, 1.0])
    light_dir = torch.tensor([0.0, 0.0, 1.0])
    light_intensity = torch.tensor([1.0, 1.0, 1.0])
    color = brdf(material, view_dir, light_dir, light_intensity)
    assert color.shape == (3, H, W)
    assert torch.all(color >= 0) and torch.all(color <= 1)


def test_cooktorrance_point():
    H, W = 64, 64
    material = DiffuseSpecularMaterial(
        albedo=torch.rand(3, H, W),
        normal=torch.rand(3, H, W) * 2 - 1,
        roughness=torch.rand(1, H, W),
        specular=torch.rand(3, H, W),
    )
    brdf = CookTorranceBRDF(light_type="point")
    view_dir = torch.tensor([0.0, 0.0, 1.0])
    light_pos = torch.tensor([0.0, 10.0, 10.0])
    light_intensity = torch.tensor([1.0, 1.0, 1.0])
    color = brdf(material, view_dir, light_pos, light_intensity, light_size=5.0)
    assert color.shape == (3, H, W)
    assert torch.all(color >= 0) and torch.all(color <= 1)


def test_lambertian_brdf():
    H, W = 32, 32
    material = MaterialBase(
        albedo=torch.rand(3, H, W),
        normal=torch.rand(3, H, W) * 2 - 1,
        roughness=torch.rand(1, H, W),
    )
    brdf = LambertianBRDF()
    view_dir = torch.tensor([0.0, 0.0, 1.0])
    light_dir = torch.tensor([0.0, 0.0, 1.0])
    light_intensity = torch.tensor([1.0, 1.0, 1.0])
    color = brdf(material, view_dir, light_dir, light_intensity)
    assert color.shape == (3, H, W)
    assert torch.all(color >= 0) and torch.all(color <= 1)


def test_orennayar_brdf():
    H, W = 32, 32
    material = MaterialBase(
        albedo=torch.rand(3, H, W),
        normal=torch.rand(3, H, W) * 2 - 1,
        roughness=torch.rand(1, H, W),
    )
    brdf = OrenNayarBRDF(sigma=0.5)
    view_dir = torch.tensor([0.0, 0.0, 1.0])
    light_dir = torch.tensor([0.0, 0.0, 1.0])
    light_intensity = torch.tensor([1.0, 1.0, 1.0])
    color = brdf(material, view_dir, light_dir, light_intensity)
    assert color.shape == (3, H, W)
    assert torch.all(color >= 0) and torch.all(color <= 1)
