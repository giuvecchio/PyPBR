import torch
import pytest
from pypbr.materials import MaterialBase
from pypbr.transforms import (
    Resize,
    RandomResize,
    Crop,
    CenterCrop,
    RandomCrop,
    Tile,
    Rotate,
    RandomRotate,
    FlipHorizontal,
    FlipVertical,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Roll,
    InvertNormal,
    AdjustNormalStrength,
    ToLinear,
    ToSrgb,
    Compose,
)


def create_dummy_material(H=32, W=32):
    return MaterialBase(
        albedo=torch.rand(3, H, W),
        normal=torch.rand(3, H, W) * 2 - 1,
        roughness=torch.rand(1, H, W),
    )


def test_resize():
    mat = create_dummy_material(32, 32)
    new_size = (64, 64)
    transform = Resize(new_size)
    new_mat = transform(mat)
    assert new_mat.size == new_size


def test_random_resize():
    mat = create_dummy_material(32, 32)
    new_mat = RandomResize(40, 80)(mat)
    H, W = new_mat.size
    assert 40 <= H <= 80 and 40 <= W <= 80


def test_crop():
    mat = create_dummy_material(32, 32)
    transform = Crop(top=5, left=5, height=20, width=20)
    new_mat = transform(mat)
    assert new_mat.size == (20, 20)


def test_center_crop():
    mat = create_dummy_material(32, 32)
    transform = CenterCrop(height=16, width=16)
    new_mat = transform(mat)
    assert new_mat.size == (16, 16)


def test_random_crop():
    mat = create_dummy_material(32, 32)
    transform = RandomCrop(height=16, width=16)
    new_mat = transform(mat)
    assert new_mat.size == (16, 16)


def test_tile():
    mat = create_dummy_material(16, 16)
    transform = Tile(num_tiles=2)
    new_mat = transform(mat)
    # Tiling should multiply the spatial dimensions
    assert new_mat.size == (16 * 2, 16 * 2)


def test_rotate():
    mat = create_dummy_material(32, 32)
    transform = Rotate(angle=90, expand=False)
    new_mat = transform(mat)
    # With expand=False, size should remain the same
    assert new_mat.size == mat.size


def test_random_rotate():
    mat = create_dummy_material(32, 32)
    transform = RandomRotate(min_angle=0, max_angle=360)
    new_mat = transform(mat)
    # After random rotation with expand=False, size remains unchanged
    assert new_mat.size == mat.size


def test_flip_horizontal():
    # Use a deterministic albedo to test horizontal flip
    H, W = 16, 16
    albedo = torch.arange(3 * H * W, dtype=torch.float32).reshape(3, H, W)
    mat = create_dummy_material(H, W)
    mat.albedo = albedo
    new_mat = FlipHorizontal()(mat)
    # Check that the albedo is equal to a horizontal flip of the original
    assert torch.allclose(new_mat.albedo, albedo.flip(-1))


def test_flip_vertical():
    H, W = 16, 16
    albedo = torch.arange(3 * H * W, dtype=torch.float32).reshape(3, H, W)
    mat = create_dummy_material(H, W)
    mat.albedo = albedo
    new_mat = FlipVertical()(mat)
    assert torch.allclose(new_mat.albedo, albedo.flip(-2))


def test_random_horizontal_flip():
    mat = create_dummy_material(16, 16)
    new_mat = RandomHorizontalFlip()(mat, p=1.0)
    # With probability 1, the material is flipped
    assert torch.allclose(new_mat.albedo, mat.albedo.flip(-1))


def test_random_vertical_flip():
    mat = create_dummy_material(16, 16)
    new_mat = RandomVerticalFlip()(mat, p=1.0)
    assert torch.allclose(new_mat.albedo, mat.albedo.flip(-2))


def test_roll():
    mat = create_dummy_material(16, 16)
    transform = Roll(shift=(2, 3))
    new_mat = transform(mat)
    # Roll doesn't change the overall size
    assert new_mat.size == mat.size


def test_invert_normal():
    mat = create_dummy_material(16, 16)
    orig_normal = mat.normal.clone()
    new_mat = InvertNormal()(mat)
    # Check that the Y channel is inverted
    assert torch.allclose(new_mat.normal[1], -orig_normal[1])


def test_to_linear_and_to_srgb():
    mat = create_dummy_material(16, 16)
    # Ensure that converting to linear and back to sRGB returns a similar albedo
    lin = ToLinear()(mat)
    srgb = ToSrgb()(lin)
    assert torch.allclose(mat.albedo, srgb.albedo, atol=1e-2)


def test_compose():
    mat = create_dummy_material(32, 32)
    transform = Compose([Resize((64, 64)), Crop(10, 10, 40, 40)])
    new_mat = transform(mat)
    assert new_mat.size == (40, 40)
