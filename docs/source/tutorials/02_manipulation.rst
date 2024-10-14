.. _material_manipulation:

Material Manipulation
=====================

In this tutorial, we will cover various techniques for manipulating materials using **PyPBR**. We'll explore how to apply transformations to materials, convert color spaces, and convert between different material workflows.

1. **Transformations**: Resize, crop, rotate, and tile texture maps.
2. **Color Space Conversions**: Convert between sRGB and linear color spaces.
3. **Workflow Conversion**: Convert between basecolor-metallic and diffuse-specular workflows.

By the end of this tutorial, you will know how to make significant modifications to materials to better suit your rendering requirements.

Applying Transformations to Materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **PyPBR** library provides several transformation methods for manipulating texture maps. These include resizing, cropping, rotating, tiling, flipping, and rolling the maps. Below are some common transformations that can be applied to materials.

Resize, Crop, Rotate, and Tile
------------------------------

.. code-block:: python

    from pypbr.material import BasecolorMetallicMaterial
    from PIL import Image

    # Load a material
    albedo_map = Image.open("path/to/albedo_map.png")
    material = BasecolorMetallicMaterial(albedo=albedo_map)

    # Resize the material texture maps
    material.resize((256, 256))

    # Crop the texture maps
    material.crop(top=10, left=10, height=200, width=200)

    # Rotate the texture maps by 45 degrees
    material.rotate(angle=45, expand=True)

    # Tile the texture maps 3 times
    material.tile(num_tiles=3)

    # Print the transformed material details
    print(material)

These transformations can be applied to all texture maps of the material simultaneously, ensuring consistency across the different channels.

Flip and Roll
-------------

You can also flip the texture maps horizontally or vertically, and roll them by a specified shift value.

.. code-block:: python

    # Flip the material texture maps horizontally
    material.flip_horizontal()

    # Roll the texture maps by (shift_height, shift_width)
    material.roll(shift=(100, 50))

Color Space Conversions
^^^^^^^^^^^^^^^^^^^^^^^

Color space conversion is important in Physically Based Rendering to ensure accurate light interactions. The **PyPBR** library supports conversion between sRGB and linear color spaces.

Convert Albedo to Linear Space
------------------------------

.. code-block:: python

    # Convert the albedo map from sRGB to linear space
    material.to_linear()

Convert Albedo to sRGB Space
----------------------------

.. code-block:: python

    # Convert the albedo map from linear to sRGB space
    material.to_srgb()

These conversions are particularly useful when working with different rendering engines that require specific color space representations.

Converting Between Workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyPBR** supports conversion between the basecolor-metallic and diffuse-specular workflows. This flexibility allows you to adapt materials to different PBR standards.

Convert Basecolor-Metallic to Diffuse-Specular
-----------------------------------------------

.. code-block:: python

    from pypbr.material import BasecolorMetallicMaterial

    # Load a diffuse-specular material
    metallic_map = Image.open("path/to/specular_map.png")
    diffuse_material = BasecolorMetallicMaterial(albedo=albedo_map, metallic=metallic_map)

    # Convert from basecolor-metallic workflow to diffuse-specular workflow
    diffuse_specular_material = material.to_diffuse_specular_material()

Convert Diffuse-Specular to Basecolor-Metallic
-----------------------------------------------

If you have a `DiffuseSpecularMaterial` and need to convert it to a `BasecolorMetallicMaterial`:

.. code-block:: python

    from pypbr.material import DiffuseSpecularMaterial

    # Load a diffuse-specular material
    specular_map = Image.open("path/to/specular_map.png")
    diffuse_material = DiffuseSpecularMaterial(albedo=albedo_map, specular=specular_map)

    # Convert to basecolor-metallic workflow
    basecolor_metallic_material = diffuse_material.to_basecolor_metallic_material()

These conversions allow you to switch between different PBR workflows, making it easier to use materials in various rendering pipelines.

Summary
^^^^^^^

In this tutorial, we explored different techniques for manipulating materials in **PyPBR**:

1. **Transformations**: Resize, crop, rotate, tile, flip, and roll texture maps.
2. **Color Space Conversions**: Convert albedo maps between sRGB and linear color spaces.
3. **Workflow Conversion**: Convert between basecolor-metallic and diffuse-specular workflows.