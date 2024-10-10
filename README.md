# PyPBR

**PyPBR** is a Python library for creating, manipulating, and rendering Physically Based Rendering (PBR) materials, with a focus on providing a modular, flexible approach to material creation and blending. 

PyPBR supports multiple PBR workflows, such as basecolor-metallic and diffuse-specular, and includes functionality for BRDF evaluation and blending operations.

PyPBR is build on PyTorch to leverage GPU acceleration and for easy integration with existing AI codebases. 

## Features

- **Material Classes**: Support for different PBR workflows, including basecolor-metallic and diffuse-specular workflows.
- **Texture Manipulation**: Resize, rotate, crop, flip, and tile texture maps to create customized materials.
- **BRDF Models**: Implementations of Bidirectional Reflectance Distribution Functions (BRDF), including the Cook-Torrance model.
- **Material Blending**: Both functional and class-based approaches for blending materials, including methods based on masks, height maps, property maps, and gradients.
- **Input/Output**: Load and save materials from/to folders, supporting common file formats for easy integration into existing workflows.

## Installation

You can install **PyPBR** via pip:

```sh
pip install pypbr
```

## Getting Started

### Creating a PBR Material

To create a PBR material, use one of the provided classes from the `pypbr.material` module:

```python
from pypbr.material import BasecolorMetallicMaterial
from PIL import Image

# Load albedo and metallic maps
albedo_image = Image.open("path/to/albedo.png")
normal_image = Image.open("path/to/normal.png")
roughness_image = Image.open("path/to/roughness.png")
metallic_image = Image.open("path/to/metallic.png")

# Create a basecolor-metallic material
material = BasecolorMetallicMaterial(
  albedo=albedo_image, 
  normal=normal_image,
  roughness=metallic_image,
  metallic=metallic_image
)

# Convert the material to a different workflow
diffuse_specular_material = material.to_diffuse_specular_material()
```

### Manipulating Texture Maps

PyPBR allows you to transform texture maps, such as resizing, rotating, and cropping:

```python
# Resize texture maps
material.resize((512, 512))

# Rotate the texture maps by 45 degrees
material.rotate(45, expand=True)

# Convert the albedo map to linear color space
material.to_linear()
```

### Evaluating BRDF

Use the `CookTorranceBRDF` class to evaluate light reflection on a material:

```python
from pypbr.models import CookTorranceBRDF
import torch

# Initialize the BRDF model
brdf = CookTorranceBRDF(light_type="point")

# Define the material and directions
view_dir = torch.tensor([0.0, 0.0, 1.0])
light_dir = torch.tensor([0.1, 0.1, 1.0])
light_intensity = torch.tensor([1.0, 1.0, 1.0])

# Evaluate the BRDF to get the reflected color
color = brdf(material, view_dir, light_dir, light_intensity)
```

### Blending Materials

You can blend two materials using different blending methods, either functionally or using class-based approaches:

```python
from pypbr.blending.functional import blend_materials
import torch

# Create two materials
material1 = load_material_from_folder("path/to/material1", preferred_workflow="metallic")
material2 = load_material_from_folder("path/to/material2", preferred_workflow="metallic")

# Blend the materials using a mask
mask = torch.rand(1, 256, 256)
blended_material = blend_materials(material1, material2, method='mask', mask=mask)
```

Or use class-based blending methods:

```python
from pypbr.blending.blending import MaskBlend

# Use a MaskBlend instance to blend two materials
mask_blend = MaskBlend(mask)
blended_material = mask_blend(material1, material2)
```

## Modules Overview

- **`pypbr.material`**: Core classes for creating and manipulating PBR materials.
- **`pypbr.models`**: Implementations of different BRDF models for rendering.
- **`pypbr.utils`**: Utility functions for color space conversions and normal map computations.
- **`pypbr.io`**: Functions for loading and saving materials.
- **`pypbr.blending.functional`**: Functional interfaces for blending materials.
- **`pypbr.blending.blending`**: Class-based blending interfaces for PBR materials.

## Contributing

Contributions are welcome! If you have ideas for new features or enhancements, feel free to open an issue or a pull request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to reach out or open an issue in the GitHub repository.

