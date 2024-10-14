.. _materials_creation:

Materials Creation
==================

This tutorial will guide you through the process of creating materials using the **PyPBR** library. 
In this tutorial we explore:

1. **Manual Creation**: Creating materials by specifying maps manually.
2. **Automatic Creation**: Loading materials from files in a folder.
3. **Customization**: Customizing materials by adding or modifying specific texture maps.

By the end of this tutorial, you'll be comfortable with creating, loading, and customizing materials for your PBR workflows.

Manual Material Creation
------------------------

The first step in using **PyPBR** is to manually create a material by defining its texture maps. In PyPBR, materials are represented by the `MaterialBase` class and its specialized subclasses.

Here is an example of manually creating a `BasecolorMetallicMaterial` using torch tensors or image files.

.. code-block:: python

    import torch
    from pypbr.material import BasecolorMetallicMaterial
    from PIL import Image

    # Define an albedo map as a PIL image
    albedo_map = Image.open("path/to/albedo_map.png")
    normal_map = Image.open("path/to/normal_map.png")

    # Define a roughness map as a tensor
    roughness_map = torch.ones((1, 512, 512))  # Placeholder roughness map

    # Create a BasecolorMetallicMaterial instance manually
    material = BasecolorMetallicMaterial(
        albedo=albedo_map,
        normal=normal_map,
        roughness=roughness_map,
        albedo_is_srgb=True
    )

    # Print the material details
    print(material)

In this example, we create a `BasecolorMetallicMaterial` by specifying an albedo map (in sRGB color space), a normal map and a roughness map. 
You can also add other maps like `normal` or `metallic` for a more detailed representation.

Automatic Material Creation (Loading from Files)
------------------------------------------------

For convenience, **PyPBR** provides a way to automatically load materials from a folder using the `load_material_from_folder` function. This is useful when you have a set of maps saved in a specific folder and want to quickly create a material without specifying each map manually.

.. code-block:: python

    from pypbr.io import load_material_from_folder

    # Define the folder path where the material maps are located
    folder_path = "path/to/material_folder"

    # Load the material from the folder
    material = load_material_from_folder(folder_path, preferred_workflow='metallic', is_srgb=True)

    # Print the loaded material
    print(material)

The `load_material_from_folder` function will scan the specified folder for common map names (e.g., `albedo`, `roughness`, `metallic`) and create an appropriate material. You can also specify a preferred workflow (`metallic` or `specular`) to guide the loading process.

Here is an example of a folder structure for loading materials:

.. code-block:: text
    
    material_folder/
    ├── albedo.png
    ├── roughness.png
    ├── metallic.png
    ├── normal.png
    └── height.png

Ensure that your folder contains the appropriate maps with recognizable names to facilitate the loading process.

Customizing Materials
---------------------

After creating a material, you may want to customize it by adding or modifying specific maps. This can be done easily by assigning new maps to the material object. You can also add custom maps that are not typically part of the standard PBR workflow.

.. code-block:: python

    # Add a custom emissive map
    emissive_map = torch.zeros((3, 512, 512))  # Placeholder emissive map
    material.emissive = emissive_map

    # Modify the roughness map
    material.roughness = torch.full((1, 512, 512), 0.5)  # Uniform roughness value

    # Print the customized material
    print(material)

Customizing materials allows you to experiment with different textures and create unique looks for your PBR renders. You can dynamically add new texture maps and manage them as attributes of the material object.

Summary
-------

In this tutorial, we covered three approaches for creating materials with **PyPBR**:

1. **Manual Creation**: Manually specifying texture maps to create a material.
2. **Automatic Creation**: Using the `load_material_from_folder` function to load materials from files.
3. **Customization**: Adding and modifying texture maps for a material.