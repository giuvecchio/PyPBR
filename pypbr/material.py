"""
pypbr.material

This module defines the MaterialBase class and its subclasses, which encapsulate various texture maps
used in Physically Based Rendering (PBR). It provides functionalities to manipulate
and convert these texture maps for rendering purposes.

Classes:
    MaterialBase: Base class representing a PBR material.
    BasecolorMetallicMaterial: Represents a PBR material using basecolor and metallic maps.
    DiffuseSpecularMaterial: Represents a PBR material using diffuse and specular maps.
"""

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from .utils import (
    compute_height_from_normal,
    compute_normal_from_height,
    invert_normal,
    linear_to_srgb,
    rotate_normals,
    srgb_to_linear,
)


class MaterialBase:
    """
    A base class representing a PBR material.

    This class provides common functionality for PBR materials, allowing for
    dynamic addition of texture maps and common operations such as resizing,
    cropping, and transforming texture maps.

    Attributes:
        albedo (torch.FloatTensor): The albedo map tensor.
        normal (torch.FloatTensor): The normal map tensor.
        roughness (torch.FloatTensor): The roughness map tensor.
    """

    # Initialization and Attribute Management
    def __init__(
        self,
        albedo: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        albedo_is_srgb: bool = True,
        normal: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        roughness: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        """
        Initialize the Material with optional texture maps.

        Args:
            albedo: The albedo map.
            albedo_is_srgb: Flag indicating if albedo is in sRGB space.
            normal: The normal map.
            roughness: The roughness map.
            device: The device to store the texture maps.
            **kwargs: Additional texture maps.
        """
        self.device = device
        self._maps = {}
        self.albedo_is_srgb = albedo_is_srgb

        # Initialize provided maps
        if albedo is not None:
            self.albedo = albedo
        if normal is not None:
            self.normal = normal
        if roughness is not None:
            self.roughness = roughness

        # Initialize additional maps
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        """
        Override __setattr__ to manage texture maps dynamically.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name in ["albedo_is_srgb", "_maps"]:
            super().__setattr__(name, value)
        elif (
            isinstance(value, (Image.Image, np.ndarray, torch.FloatTensor))
            or value is None
        ):
            # Process and store texture maps in the _maps dictionary
            self._maps[name] = self._process_map(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        """
        Override __getattr__ to retrieve texture maps from the _maps dictionary.

        Args:
            name: Attribute name.

        Returns:
            The texture map tensor associated with the given name.
        """
        if name in self._maps:
            return self._maps[name]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def _to_tensor(
        self, image: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]]
    ) -> Optional[torch.FloatTensor]:
        """
        Convert an image to a torch tensor.

        Args:
            image: The image to convert.

        Returns:
            torch.FloatTensor: The image as a tensor.

        Raises:
            TypeError: If the input image type is unsupported.
        """
        if image is None:
            return None
        if isinstance(image, torch.FloatTensor):
            return image.to(self.device)
        elif isinstance(image, np.ndarray):
            return torch.from_numpy(image).float().to(self.device)
        elif isinstance(image, Image.Image):
            # Handle different image modes
            if image.mode in ["I", "I;16", "I;16B", "I;16L", "I;16N"]:
                # Convert 16-bit image to NumPy array
                np_image = np.array(image, dtype=np.uint16)
                tensor = torch.from_numpy(np_image.astype(np.float32))
                tensor = tensor.unsqueeze(0)  # Add channel dimension
                # Normalize to [0, 1] range
                tensor = tensor / 65535.0
                return tensor.to(self.device)
            elif image.mode == "F":
                # 32-bit floating point image
                np_image = np.array(image, dtype=np.float32)
                tensor = torch.from_numpy(np_image)
                tensor = tensor.unsqueeze(0)  # Add channel dimension
                return tensor.to(self.device)
            else:
                # For other modes, use torchvision transforms
                return TF.to_tensor(image).to(self.device)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. Supported types are PIL.Image.Image, np.ndarray, and torch.FloatTensor."
            )

    def _process_map(self, name, value):
        """
        Process the input value and convert it to a tensor if necessary.

        Args:
            name: Name of the texture map.
            value: The texture map value.

        Returns:
            torch.FloatTensor: The processed texture map tensor.
        """
        if value is None:
            return None
        tensor = self._to_tensor(value)
        if name == "normal":
            return self._process_normal_map(tensor)
        elif name == "albedo":
            return tensor
        else:
            return tensor

    def _process_normal_map(
        self, normal_map: Optional[torch.FloatTensor]
    ) -> Optional[torch.FloatTensor]:
        """
        Process the normal map by computing the Z-component if necessary and normalizing.

        Args:
            normal_map: The normal map tensor.

        Returns:
            torch.FloatTensor: The processed normal map.
        """
        if normal_map is None:
            return None

        if normal_map.shape[0] == 2:
            # Compute the Z-component
            normal_map = self._compute_normal_map_z_component(normal_map)

        elif normal_map.shape[0] == 3:
            # Check if the normal map is already normalized
            if normal_map.min() < 0:
                return normal_map

            # Convert from [0,1] to [-1,1]
            normal_map = normal_map * 2.0 - 1.0
            normal_map = F.normalize(normal_map, dim=0)
        else:
            raise ValueError("Normal map must have 2 or 3 channels.")

        return normal_map

    def _compute_normal_map_z_component(
        self, normal_xy: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Compute the Z-component of the normal map from the X and Y components.

        Args:
            normal_xy: A tensor containing the X and Y components of the normal map.

        Returns:
            torch.FloatTensor: The normal map tensor with X, Y, and Z components.
        """
        normal_xy = normal_xy * 2 - 1  # Scale from [0,1] to [-1,1]
        x = normal_xy[0:1]
        y = normal_xy[1:2]
        squared = x**2 + y**2
        z = torch.sqrt(torch.clamp(1.0 - squared, min=0.0))
        normal = torch.cat([x, y, z], dim=0)
        normal = F.normalize(normal, dim=0)
        return normal.to(self.device)

    # Device Management
    def to(self, device: torch.device):
        """
        Moves all tensors in the material to the specified device.

        Args:
            device (torch.device): The target device.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        self.device = device
        for name, map_value in self._maps.items():
            if map_value is not None:
                self._maps[name] = map_value.to(device)
        return self

    # Properties
    @property
    def linear_albedo(self):
        """
        Get the albedo map in linear space.

        Returns:
            torch.FloatTensor: The albedo map in linear space.
        """
        albedo = self._maps.get("albedo", None)
        if albedo is not None:
            if self.albedo_is_srgb:
                return srgb_to_linear(albedo)
            else:
                return albedo
        else:
            return None

    @property
    def normal_rgb(self):
        """
        Get the normal map in RGB space.

        Returns:
            torch.FloatTensor: The normal map in RGB space.
        """
        normal = self._maps.get("normal", None)
        if normal is not None:
            return (normal + 1.0) * 0.5
        else:
            return None

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """
        Get the size of the texture maps.

        Returns:
            Optional[Tuple[int, int]]: A tuple (height, width) representing the size of the texture maps.
            If multiple maps are present, returns the size of the first non-None map.
            Returns None if no maps are available.
        """
        for map_value in self._maps.values():
            if map_value is not None:
                _, height, width = map_value.shape
                return (height, width)
        return None

    # Transformation Methods
    def resize(self, size: Union[int, Tuple[int, int]], antialias: bool = True):
        """
        Resize all texture maps to the specified size.

        Args:
            size: The desired output size.
            antialias: Whether to apply antialiasing.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        for name, map_value in self._maps.items():
            if map_value is not None:
                self._maps[name] = TF.resize(map_value, size, antialias=antialias)
        return self

    def crop(self, top: int, left: int, height: int, width: int):
        """
        Crop all texture maps to the specified region.

        Args:
            top: The top pixel coordinate.
            left: The left pixel coordinate.
            height: The height of the crop.
            width: The width of the crop.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        for name, map_value in self._maps.items():
            if map_value is not None:
                self._maps[name] = TF.crop(map_value, top, left, height, width)
        return self

    def tile(self, num_tiles: int):
        """
        Tile all texture maps by repeating them.

        Args:
            num_tiles: Number of times to tile the textures.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        for name, map_value in self._maps.items():
            if map_value is not None:
                self._maps[name] = map_value.repeat(1, num_tiles, num_tiles)
        return self

    def rotate(
        self,
        angle: float,
        expand: bool = False,
        padding_mode: str = "constant",
    ):
        """
        Rotate all texture maps by a given angle.

        Args:
            angle (float): The rotation angle in degrees.
            expand (bool): Whether to expand the output image to hold the entire rotated image.
            padding_mode (str): Padding mode. Options are 'constant' or 'circular'.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        assert padding_mode in [
            "constant",
            "circular",
        ], "Invalid padding mode. Must be 'constant' or 'circular'."

        for name, map_value in self._maps.items():
            if map_value is not None:
                # Get the height and width of the image (Assuming map_value shape is (C, H, W))
                height, width = map_value.shape[-2:]

                # Convert the rotation angle from degrees to radians
                angle_rad = math.radians(angle)

                # Determine the target size after rotation
                if expand:
                    # When expanding, we compute the new size required to fit the rotated image
                    new_width = math.ceil(
                        abs(width * math.cos(angle_rad))
                        + abs(height * math.sin(angle_rad))
                    )
                    new_height = math.ceil(
                        abs(width * math.sin(angle_rad))
                        + abs(height * math.cos(angle_rad))
                    )
                    height, width = new_height, new_width

                # Compute symmetric padding amounts
                padded_size = math.ceil(math.sqrt(height**2 + width**2))
                pad_size = padded_size - height

                # Pad the image
                map_value = F.pad(
                    map_value, (pad_size, pad_size, pad_size, pad_size), padding_mode
                )

                # Rotate the padded image
                rotated_map = TF.rotate(map_value, angle, expand=True)

                # Crop the rotated image to the target size
                rotated_map = TF.center_crop(rotated_map, (height, width)).contiguous()

                if name == "normal":
                    # Rotate the normal vectors
                    rotated_map = rotate_normals(rotated_map, angle)

                self._maps[name] = rotated_map

        return self

    def flip_horizontal(self):
        """Flip all texture maps horizontally.
        When flipping normal maps, the X component is inverted.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        for name, map_value in self._maps.items():
            if map_value is not None:
                # Flip the map horizontally
                flipped_map = map_value.flip(-1)  # Flip along the width dimension
                if name == "normal":
                    # Invert the X component of the normal map
                    flipped_map = flipped_map.clone()
                    flipped_map[0] = -flipped_map[0]
                self._maps[name] = flipped_map
        return self

    def flip_vertical(self):
        """Flip all texture maps vertically.
        When flipping the normal map, the Y component is inverted.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        for name, map_value in self._maps.items():
            if map_value is not None:
                # Flip the map vertically
                flipped_map = map_value.flip(-2)  # Flip along the height dimension
                if name == "normal":
                    # Invert the Y component of the normal map
                    flipped_map = flipped_map.clone()
                    flipped_map[1] = -flipped_map[1]
                self._maps[name] = flipped_map
        return self

    def roll(self, shift: Tuple[int, int]):
        """
        Roll all texture maps along the specified shift dimensions.

        Args:
            shift: The shift values for each dimension.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        for name, map_value in self._maps.items():
            if map_value is not None:
                rolled_map = F.roll(map_value, shift, dims=(1, 2))
                self._maps[name] = rolled_map
        return self

    def apply_transform(self, transform):
        """
        Apply a transformation to all texture maps.

        Args:
            transform: A function that takes a tensor and returns a tensor.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        for name, map_value in self._maps.items():
            if map_value is not None:
                self._maps[name] = transform(map_value)
        return self

    # Normal Map Operations
    def invert_normal(self):
        """
        Invert the Y component of the normal map.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        normal = self._maps.get("normal", None)
        self._maps["normal"] = invert_normal(normal)
        return self

    def adjust_normal_strength(self, strength_factor: float):
        """Adjust the strength of the normal map.

        Args:
            strength_factor (float): The factor to adjust the strength of the normal map.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        if self.normal is not None:
            # Ensure the normal map is in [-1, 1]
            normal = self.normal
            # Adjust the X and Y components
            normal[:2] *= strength_factor
            # Re-normalize the normal vector
            normal = F.normalize(normal, dim=0)
            self._maps["normal"] = normal
        return self

    def compute_normal_from_height(self, scale: float = 1.0):
        """
        Compute the normal map from the height map.

        Args:
            scale (float): The scaling factor for the height map gradients.
                            Controls the strength of the normals.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        height_map = self._maps.get("height", None)

        # Compute the normal map from the height map
        normal_map = compute_normal_from_height(height_map, scale)

        # Store the normal map
        self._maps["normal"] = normal_map

        return self

    def compute_height_from_normal(self, scale: float = 1.0):
        """
        Compute the height map from the normal map using Poisson reconstruction.

        Args:
            scale (float): Scaling factor for the gradients.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        normal_map = self._maps.get("normal", None)

        # Compute the height map from the normal map
        height_map = compute_height_from_normal(normal_map, scale)

        # Store the height map
        self._maps["height"] = height_map

        return self

    # Color Space Conversion
    def to_linear(self):
        """
        Convert the albedo map to linear space if it's in sRGB.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        albedo = self._maps.get("albedo", None)
        if albedo is not None and self.albedo_is_srgb:
            self._maps["albedo"] = srgb_to_linear(albedo)
            self.albedo_is_srgb = False
        return self

    def to_srgb(self):
        """
        Convert the albedo map to sRGB space if it's in linear space.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        albedo = self._maps.get("albedo", None)
        if albedo is not None and not self.albedo_is_srgb:
            self._maps["albedo"] = linear_to_srgb(albedo)
            self.albedo_is_srgb = True
        return self

    # Conversion Methods
    def to_numpy(self):
        """
        Convert all texture maps to NumPy arrays.

        Returns:
            dict: A dictionary containing NumPy arrays of the texture maps.
        """
        maps = {}
        for name, map_value in self._maps.items():
            maps[name] = map_value.cpu().numpy() if map_value is not None else None
        return maps

    def to_pil(
        self,
        maps_mode: Dict[str, str] = None,
    ):
        """
        Convert all texture maps to PIL Images.

        Args:
            maps_mode (Dict[str, str]): Optional dictionary specifying modes for each map type.

        Returns:
            dict: A dictionary containing PIL Images of the texture maps.
        """
        # Default maps_mode if not provided
        if maps_mode is None:
            maps_mode = {}

        maps = {}
        for name, map_value in self._maps.items():
            if map_value is not None:
                if name == "normal":
                    # Scale the normal map from [-1, 1] to [0, 1] before converting to PIL
                    map_value = (map_value + 1.0) * 0.5

                # Determine the mode for this map, default to 'RGB'
                mode = maps_mode.get(name, "RGB")

                # Convert the tensor to a PIL Image
                map_value = map_value.cpu()

                if mode in ["I", "I;16", "I;16B", "I;16L", "I;16N"]:
                    # Handle 16-bit image conversion using numpy
                    # Convert the image to numpy array
                    image_np = map_value.numpy()

                    # Scale image to 16-bit
                    image_np = (image_np * 65535).astype(np.uint16)

                    # Create a new PIL Image from the numpy array in mode 'I;16'
                    maps[name] = Image.fromarray(image_np, mode="I;16")
                else:
                    # Determine the number of channels in the image
                    mode = "RGB" if map_value.shape[0] == 3 else "L"

                    # Ensure the image is in the desired mode
                    image = TF.to_pil_image(map_value.cpu())
                    maps[name] = image.convert(mode)
            else:
                maps[name] = None
        return maps

    # Utility Methods
    def __repr__(self):
        """
        Return a string representation of the Material object.

        Returns:
            str: String representation of the Material.
        """
        repr_str = f"{self.__class__.__name__}("
        attrs = [
            f"{name}={getattr(self, name).shape if getattr(self, name) is not None else None}"
            for name in self._maps.keys()
        ]
        repr_str += ", ".join(attrs)
        repr_str += ")"
        return repr_str

    def save_to_folder(self, folder_path: str):
        """
        Save the material maps to a folder.

        Args:
            folder_path: The path to the folder where maps will be saved.
        """
        from .io import save_material_to_folder

        save_material_to_folder(self, folder_path)


class BasecolorMetallicMaterial(MaterialBase):
    """
    A class representing a PBR material using basecolor and metallic maps.

    Attributes:
        albedo (torch.FloatTensor): The albedo map tensor.
        normal (torch.FloatTensor): The normal map tensor.
        roughness (torch.FloatTensor): The roughness map tensor.
        metallic (torch.FloatTensor): The metallic map tensor.
    """

    def __init__(
        self,
        albedo: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        albedo_is_srgb: bool = True,
        normal: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        roughness: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        metallic: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        **kwargs,
    ):
        """
        Initialize the Material with optional texture maps.

        Args:
            albedo: The albedo map.
            albedo_is_srgb: Flag indicating if albedo is in sRGB space.
            normal: The normal map.
            roughness: The roughness map.
            metallic: The metallic map.
            **kwargs: Additional texture maps.
        """
        super().__init__(
            albedo=albedo,
            albedo_is_srgb=albedo_is_srgb,
            normal=normal,
            roughness=roughness,
            **kwargs,
        )
        if metallic is not None:
            self.metallic = metallic

    def to_diffuse_specular_material(self, albedo_is_srgb: bool = False):
        """
        Convert the material from basecolor-metallic workflow to diffuse-specular workflow.

        Args:
            albedo_is_srgb: Flag indicating if the albedo map should be returned in sRGB space.

        Returns:
            DiffuseSpecularMaterial: A new material instance in the diffuse-specular workflow.
        """
        # Ensure albedo and metallic maps are available
        if self.albedo is None or self.metallic is None:
            raise ValueError(
                "Both albedo and metallic maps are required for conversion."
            )

        # Convert albedo to linear space if necessary
        albedo = self.linear_albedo

        # Resize metallic map to match albedo size if necessary
        if self.metallic.shape[1:] != albedo.shape[1:]:
            metallic = TF.resize(self.metallic, albedo.shape[1:])
        else:
            metallic = self.metallic

        # Ensure metallic map has the correct dimensions
        if metallic.dim() == 2:
            metallic = metallic.unsqueeze(0)  # Add channel dimension

        # Specular dielectric color (F0 for non-metals)
        specular_dielectric = torch.full_like(albedo, 0.04)  # 4% reflectance

        # Compute diffuse color
        diffuse = albedo * (1.0 - metallic)

        # Compute specular color
        specular = specular_dielectric * (1.0 - metallic) + albedo * metallic

        # Create a new DiffuseSpecularMaterial instance
        diffuse_specular_material = DiffuseSpecularMaterial(
            albedo=diffuse,
            specular=specular,
            normal=self.normal,
            roughness=self.roughness,
            albedo_is_srgb=albedo_is_srgb,  # The diffuse map is in linear space
        )

        return diffuse_specular_material


class DiffuseSpecularMaterial(MaterialBase):
    """
    A class representing a PBR material using diffuse and specular maps.

    Attributes:
        albedo (torch.FloatTensor): The albedo map tensor.
        normal (torch.FloatTensor): The normal map tensor.
        roughness (torch.FloatTensor): The roughness map tensor.
        specular (torch.FloatTensor): The specular map tensor.
    """

    def __init__(
        self,
        albedo: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        albedo_is_srgb: bool = True,
        normal: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        roughness: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        specular: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        specular_is_srgb: bool = True,
        **kwargs,
    ):
        """
        Initialize the Material with optional texture maps.

        Args:
            albedo: The albedo map.
            albedo_is_srgb: Flag indicating if albedo is in sRGB space.
            normal: The normal map.
            roughness: The roughness map.
            specular: The specular map.
            specular_is_srgb: Flag indicating if specular is in sRGB space.
            **kwargs: Additional texture maps.
        """
        super().__init__(
            albedo=albedo,
            albedo_is_srgb=albedo_is_srgb,
            normal=normal,
            roughness=roughness,
            **kwargs,
        )

        self.specular_is_srgb = specular_is_srgb
        if specular is not None:
            self.specular = specular

    def to_basecolor_metallic_material(self, albedo_is_srgb: bool = False):
        """
        Convert the material from diffuse-specular workflow to basecolor-metallic workflow.

        Args:
            albedo_is_srgb: Flag indicating if the albedo map should be returned in sRGB space.

        Returns:
            BasecolorMetallicMaterial: A new material instance in the basecolor-metallic workflow.
        """
        # Ensure diffuse and specular maps are available
        if self.albedo is None or self.specular is None:
            raise ValueError(
                "Both albedo (diffuse) and specular maps are required for conversion."
            )

        # Convert albedo (diffuse) to linear space if necessary
        diffuse = self.linear_albedo

        # Specular dielectric color (F0 for non-metals)
        specular_dielectric = 0.04  # 4% reflectance

        # Resize specular map to match diffuse size if necessary
        if self.specular.shape[1:] != diffuse.shape[1:]:
            specular = TF.resize(self.specular, diffuse.shape[1:], antialias=True)
        else:
            specular = self.specular

        # Ensure specular map has the correct dimensions
        if specular.dim() == 2:
            specular = specular.unsqueeze(0)  # Add channel dimension

        # Calculate metallic map
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-6
        numerator = specular - specular_dielectric
        denominator = diffuse - specular_dielectric + epsilon
        metallic = torch.clamp(numerator / (denominator + epsilon), 0.0, 1.0)

        # Handle edge cases where denominator is very small
        metallic = torch.where(
            denominator < epsilon, torch.zeros_like(metallic), metallic
        )

        # Compute basecolor
        basecolor = (diffuse) / (1.0 - metallic + epsilon)

        # Where metallic is close to 1, use specular as basecolor
        metallic_threshold = 0.95
        basecolor = torch.where(metallic >= metallic_threshold, specular, basecolor)

        # Clamp basecolor to valid range
        basecolor = torch.clamp(basecolor, 0.0, 1.0)

        # Create a new BasecolorMetallicMaterial instance
        basecolor_metallic_material = BasecolorMetallicMaterial(
            albedo=basecolor,
            metallic=metallic,
            normal=self.normal,
            roughness=self.roughness,
            albedo_is_srgb=albedo_is_srgb,  # The basecolor is in linear space
        )

        return basecolor_metallic_material

    @property
    def linear_specular(self):
        """
        Get the specular map in linear space.

        Returns:
            torch.FloatTensor: The albedo map in linear space.
        """
        specular = self._maps.get("specular", None)
        if specular is not None:
            if self.specular_is_srgb:
                return srgb_to_linear(specular)
            else:
                return specular
        else:
            return None

    def to_linear(self):
        """
        Convert the albedo and specular maps to linear space if it's in sRGB.

        Returns:
            MaterialBase: Returns self for method chaining.
        """

        super().to_linear()

        specular = self._maps.get("specular", None)
        if specular is not None and self.specular_is_srgb:
            self._maps["specular"] = srgb_to_linear(specular)
            self.specular_is_srgb = False

        return self

    def to_srgb(self):
        """
        Convert the albedo and specular maps to sRGB space if it's in linear space.

        Returns:
            MaterialBase: Returns self for method chaining.
        """
        super().to_srgb()
        specular = self._maps.get("specular", None)
        if specular is not None and not self.specular_is_srgb:
            self._maps["specular"] = linear_to_srgb(specular)
            self.specular_is_srgb = True

        return self
