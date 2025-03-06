"""
pypbr.materials.base

This module defines the `MaterialBase` class, which encapsulate various texture maps
used in Physically Based Rendering (PBR). It provides functionalities to manipulate
and convert these texture maps for rendering purposes.

Classes:
    `MaterialBase`: Base class representing a PBR material.
"""

import copy
import math
from collections.abc import Iterable, Mapping
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from ..utils import (
    NormalConvention,
    compute_height_from_normal,
    compute_normal_from_height,
    invert_normal,
    linear_to_srgb,
    rotate_normals,
    srgb_to_linear,
)


class MaterialBase:
    """
    Base Class for PBR Materials.

    This class provides common functionality for PBR materials, allowing for
    dynamic addition of texture maps and common operations such as resizing,
    cropping, and transforming texture maps.

    Attributes:
        albedo (torch.FloatTensor): The albedo map tensor.
        normal (torch.FloatTensor): The normal map tensor.
        roughness (torch.FloatTensor): The roughness map tensor.
    """

    def __init__(
        self,
        albedo: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        albedo_is_srgb: bool = True,
        normal: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        roughness: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        normal_convention: NormalConvention = NormalConvention.OPENGL,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        """
        Initialize the Material with optional texture maps.

        Args:
            albedo (Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]]): The albedo map.
            albedo_is_srgb (bool): Flag indicating if albedo is in sRGB space. Defaults to True.
            normal (Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]]): The normal map.
            roughness (Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]]): The roughness map.
            device (torch.device): The device to store the texture maps. Defaults to CPU.
            **kwargs: Additional texture maps.
        """
        self.device = device
        self.normal_convention = normal_convention
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
                if image.mode == "RGBA":
                    # Convert RGBA to RGB
                    image = image.convert("RGB")
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

    # Dictionary and Tensor Methods
    def as_dict(self) -> Dict[str, torch.FloatTensor]:
        """
        Get all texture maps as a dictionary.

        Returns:
            dict: A dictionary containing all texture maps.
        """
        return {name: map_value for name, map_value in self._maps.items()}

    def as_tensor(
        self,
        names: Optional[List[Union[str, Tuple[str, int]]]] = None,
        normalize: Optional[bool] = False,
    ) -> torch.FloatTensor:
        """
        Get a subset of texture maps stacked in a tensor.

        Args:
            names (Optional[List[Union[str, Tuple[str, int]]]]):
                - If None or empty, include all maps with all channels.
                - If a list of strings, include only those maps with all channels.
                - If a list of tuples, each tuple contains:
                    - map name (str)
                    - number of channels to include (int)
                - The list can contain a mix of strings and tuples.
            normalize (Optional[bool]): Wether to normalized in range [-1, 1].

        Returns:
            torch.FloatTensor: A tensor containing the specified texture maps stacked along the channel dimension.

        Raises:
            KeyError: If a specified map name does not exist in self._maps.
            ValueError: If the requested number of channels exceeds the available channels in a map.
            TypeError: If 'names' is not of the expected type.
        """
        # Build a list of (map_name, channel_limit) tuples
        selected_maps: List[Tuple[str, Optional[int]]] = []

        if not names:
            # Include all maps with all channels
            selected_maps = [(name, None) for name in self._maps.keys()]
        else:
            if not isinstance(names, list):
                raise TypeError("names must be a list of strings or tuples.")

            for item in names:
                if isinstance(item, str):
                    # Include all channels for this map
                    selected_maps.append((item, None))
                elif isinstance(item, tuple):
                    if len(item) != 2:
                        raise ValueError(
                            "Each tuple in names must have exactly two elements: (map_name, channel_limit)."
                        )
                    map_name, channel_limit = item
                    if not isinstance(map_name, str):
                        raise TypeError(
                            "The first element of each tuple must be a string (map name)."
                        )
                    if not isinstance(channel_limit, int) or channel_limit <= 0:
                        raise ValueError(
                            "The second element of each tuple must be a positive integer (channel limit)."
                        )
                    selected_maps.append((map_name, channel_limit))
                else:
                    raise TypeError(
                        "Each item in names must be either a string or a tuple of (str, int)."
                    )

        tensors = []
        for name, channel_limit in selected_maps:
            if name not in self._maps:
                raise KeyError(f"Map '{name}' does not exist in the texture maps.")

            tensor = self._maps[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Map '{name}' is not a torch.Tensor.")

            if channel_limit is not None:
                available_channels = tensor.size(0)
                if channel_limit > available_channels:
                    raise ValueError(
                        f"Requested {channel_limit} channels for map '{name}', "
                        f"but only {available_channels} channels are available."
                    )
                tensor = tensor[:channel_limit]

            if normalize and name != "normal":
                tensor = (tensor - 0.5) / 0.5

            tensors.append(tensor)

        if not tensors:
            raise ValueError("No valid texture maps found to stack.")

        # Ensure all tensors have the same spatial dimensions
        spatial_dims = [tensor.shape[1:] for tensor in tensors]
        if not all(dim == spatial_dims[0] for dim in spatial_dims):
            raise ValueError(
                "All texture maps must have the same spatial dimensions for concatenation."
            )

        # Concatenate tensors along the channel dimension
        stacked_tensor = torch.cat(tensors, dim=0)
        return stacked_tensor

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.FloatTensor,
        names: Optional[List[Union[str, Tuple[str, int]]]] = None,
    ) -> "MaterialBase":
        """
        Create a new MaterialBase instance by unpacking a tensor into texture maps.

        Args:
            tensor (torch.FloatTensor): A packed tensor of shape (C_total, H, W).
            names (list): List specifying the order and channel count for each map.
                          For example: [("albedo", 3), ("normal", 3), ("roughness", 1)]

        Returns:
            An instance of MaterialBase (or a subclass) with its _maps populated.
        """
        # Create a new instance of the class.
        instance = cls()

        # If no configuration is provided, we assume default ordering from instance._maps.
        if not names:
            names = [
                (name, instance._maps[name].size(0)) for name in instance._maps.keys()
            ]

        # Determine the total number of channels expected.
        total_channels_expected = 0
        config = []
        for item in names:
            if isinstance(item, str):
                if item in instance._maps and isinstance(
                    instance._maps[item], torch.Tensor
                ):
                    channels = instance._maps[item].size(0)
                else:
                    raise KeyError(
                        f"Cannot infer channel count for map '{item}'. Provide a tuple instead."
                    )
                config.append((item, channels))
                total_channels_expected += channels
            elif isinstance(item, tuple):
                if len(item) != 2:
                    raise ValueError("Each tuple must be (map_name, channel_limit).")
                map_name, channel_limit = item
                config.append((map_name, channel_limit))
                total_channels_expected += channel_limit
            else:
                raise TypeError(
                    "Configuration items must be a string or tuple (str, int)."
                )

        if tensor.size(0) != total_channels_expected:
            raise ValueError(
                f"Packed tensor has {tensor.size(0)} channels, but configuration expects {total_channels_expected} channels."
            )

        # Unpack the tensor along the channel dimension.
        index = 0
        for map_name, num_channels in config:
            instance._maps[map_name] = tensor[index : index + num_channels].clone()
            index += num_channels

        return instance

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
                rolled_map = torch.roll(map_value, shift, dims=(1, 2))
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
        self.normal_convention = (
            NormalConvention.DIRECTX
            if self.normal_convention == NormalConvention.OPENGL
            else NormalConvention.OPENGL
        )
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
        normal_map = compute_normal_from_height(
            height_map, scale, convention=self.normal_convention
        )

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
        height_map = compute_height_from_normal(
            normal_map, scale, convention=self.normal_convention
        )

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
                    if map_value.shape[0] == 2:
                        # Compute the Z-component
                        map_value = self._compute_normal_map_z_component(map_value)
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
        from ..io import save_material_to_folder

        save_material_to_folder(self, folder_path)

    def clone(self):
        """
        Create a deep copy of the Material object.

        Returns:
            MaterialBase: A deep copy of the Material object.
        """

        def clone_obj(obj):
            if isinstance(obj, torch.Tensor):
                # Clone the tensor
                return obj.clone()
            elif isinstance(obj, Mapping):
                # If it's a dict-like object, process each key-value pair
                return {k: clone_obj(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                # If it's a list, tuple, or set, process each element
                if isinstance(obj, list):
                    return [clone_obj(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(clone_obj(item) for item in obj)
                elif isinstance(obj, set):
                    return {clone_obj(item) for item in obj}
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                # For other iterable types, you can handle them as needed
                # Here, we'll attempt to clone each element
                return type(obj)(clone_obj(item) for item in obj)
            else:
                # For non-tensor and non-container objects, make a shallow copy
                return copy.copy(obj)

        new_dict = {k: clone_obj(v) for k, v in self.__dict__.items()}
        return self.__class__(**new_dict)
