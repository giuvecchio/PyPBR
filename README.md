# PyPBR

PyPBR is a Python library for easy and fast manipulation of Physically Based Rendering (PBR) materials with PyTorch integration. It simplifies the process of loading, manipulating, and saving PBR materials, and provides tools for working with BRDF models.

## Features

- Easy loading and saving of PBR materials from folders with naming conventions.
- Integration with PyTorch tensors for seamless GPU acceleration.
- Material manipulation functions (resize, tile, invert normals, etc.).

## Installation

```bash
pip install pypbr
```

## Usage

```python
from pypbr import Material, load_material_from_folder

# Load a material from a folder
material = load_material_from_folder('path/to/material/folder')

# Resize the material
material.resize((512, 512))

# Invert the normal map
material.invert_normal()

# Save the modified material
material.save_to_folder('path/to/save/material')
```