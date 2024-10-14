.. _getting-started:

Getting Started
============================

Welcome to the **Getting Started** guide for PyPBR! This guide will help you set up PyPBR and start creating and manipulating Physically Based Rendering (PBR) materials in no time.

Installation
------------

To install PyPBR, use `pip` from your command line:

.. code-block:: sh

   pip install pypbr

Ensure that you have Python 3.6 or higher installed, along with `torch` for PyTorch operations, which is a key dependency for PyPBR.

Basic Setup
-----------

To begin using PyPBR, you can import the necessary modules and create your first material. Below is a basic example of creating a material using the `BasecolorMetallicMaterial` class:

.. code-block:: python

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

   # Print material representation
   print(material)

Loading Materials
----------------------------

PyPBR allows you to load materials from a folder and supports different workflows and high customization. 
Here’s how you can load a basecolor-metallic material:

.. code-block:: python

   from pypbr.io import load_material_from_folder
   import torch

   # Load the material
   material = load_material_from_folder("/path/to/material", preferred_workflow="metallic")

   # Convert to a different workflow if needed
   diffuse_specular_material = material.to_diffuse_specular_material()

Manipulating Texture Maps
-------------------------

PyPBR provides various utilities to manipulate texture maps. You can resize, crop, rotate, or flip these maps as required.

.. code-block:: python

   # Resize the texture maps to 256x256
   material.resize((256, 256))

   # Rotate texture maps by 45 degrees
   material.rotate(45, expand=True)

   # Crop the texture maps to a specific region
   material.crop(0, 0, 128, 128)

Evaluating Material BRDF
------------------------

To evaluate the reflection properties of a material, you can use PyPBR’s BRDF models, such as the `CookTorranceBRDF`. Below is an example:

.. code-block:: python

   from pypbr.models import CookTorranceBRDF

   # Initialize the BRDF model
   brdf = CookTorranceBRDF(light_type="point")

   # Define the view direction, light direction, and light intensity
   view_dir = torch.tensor([0.0, 0.0, 1.0])
   light_dir = torch.tensor([0.1, 0.1, 1.0])
   light_intensity = torch.tensor([1.0, 1.0, 1.0])
   light_size = 1.0

   # Evaluate the BRDF
   reflected_color = brdf(material, view_dir, light_dir, light_intensity, light_size)

Blending Materials
------------------

PyPBR also includes functionalities for blending different materials using a variety of methods, such as masks, height maps, or gradients.

.. code-block:: python

   from pypbr.blending.functional import blend_materials
   import torch

   # Create two materials to blend
   material1 = BasecolorMetallicMaterial(albedo=torch.rand(3, 256, 256))
   material2 = BasecolorMetallicMaterial(albedo=torch.rand(3, 256, 256))

   # Define a blending mask
   mask = torch.rand(1, 256, 256)

   # Blend the two materials using the mask
   blended_material = blend_materials(material1, material2, method='mask', mask=mask)