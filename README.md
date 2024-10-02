# PyPBR

<!-- ![PyPBR Logo](https://github.com/giuvecchio/PyPBR/raw/main/assets/logo.png) -->

PyPBR is a versatile Python library designed for easy and efficient manipulation of Physically Based Rendering (PBR) materials with seamless PyTorch integration. It simplifies loading, converting, manipulating, and saving PBR materials, and offers robust tools for working with advanced Bidirectional Reflectance Distribution Function (BRDF) models like Cook-Torrance.

## Features

- **Multiple PBR Workflows:**
  - **Basecolor-Metallic** and **Diffuse-Specular** workflows supported.
  - Seamless conversion between workflows with dedicated methods.

- **Dynamic Material Handling:**
  - Automated selection of appropriate material classes (`BasecolorMetallicMaterial` or `DiffuseSpecularMaterial`) based on available texture maps using the Factory pattern.

- **High-Precision Map Support:**
  - Support for high bit-depth maps such as 16-bit height maps without loss of precision.

- **Color Space Management:**
  - Proper handling of sRGB and linear color spaces for albedo, diffuse, and specular maps.
  - Flags to specify color space properties for each map.

- **PyTorch Integration:**
  - Utilizes PyTorch tensors for all material maps, enabling GPU acceleration and efficient computations.

- **Material Manipulation Tools:**
  - Functions to resize, tile, invert normals, and perform other common material manipulations.

- **Advanced BRDF Models:**
  - Implementation of the **Cook-Torrance** BRDF model that supports multiple workflows and ensures consistent rendering results.

- **Flexible Input/Output:**
  - Easy loading and saving of PBR materials from and to folders using standardized naming conventions.
  - Support for various image formats including PNG, JPEG, TIFF, BMP, and EXR.

## Installation

PyPBR can be installed via pip:

```bash
pip install pypbr
```

Ensure that you have PyTorch installed. If not, follow the [official installation guide](https://pytorch.org/get-started/locally/) to set it up.

## Quick Start

### Loading and Manipulating Materials

```python
from pypbr.io import load_material_from_folder, save_material_to_folder

# Load a material from a folder with automatic workflow detection
material = load_material_from_folder('path/to/material/folder', preferred_workflow='metallic')

# Convert the material to the diffuse-specular workflow
diffuse_specular_material = material.to_diffuse_specular_material()

# Manipulate the material
diffuse_specular_material.resize((512, 512))
diffuse_specular_material.invert_normal()

# Save the modified material to a new folder
save_material_to_folder(diffuse_specular_material, 'path/to/save/material', format='png')
```

### Rendering with Cook-Torrance BRDF

```python
import torch
from pypbr.io import load_material_from_folder
from pypbr.models import CookTorranceBRDF

# Load a material (can be BasecolorMetallicMaterial or DiffuseSpecularMaterial)
material = load_material_from_folder('path/to/material/folder')

# Initialize the BRDF model
brdf = CookTorranceBRDF(light_type='directional')

# Define view and light directions
view_dir = torch.tensor([0.0, 0.0, 1.0])  # Viewer looking straight at the surface
light_dir = torch.tensor([0.0, 0.0, 1.0])  # Light coming from the same direction

# Define light intensity
light_intensity = torch.tensor([1.0, 1.0, 1.0])  # White light

# Compute the reflected color
color = brdf(
    material=material,
    view_dir=view_dir,
    light_dir_or_position=light_dir,
    light_intensity=light_intensity
)

# color is a tensor with shape (3, H, W) representing the RGB values
```