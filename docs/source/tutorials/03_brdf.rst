.. _brdf_evaluation:

BRDF Evaluation and Rendering
==============================

In this tutorial, we will cover how to use the **PyPBR** library to evaluate Bidirectional Reflectance Distribution Functions (BRDFs) and perform rendering. 
The BRDF evaluation in **PyPBR** is implemented to facilitate the realistic representation of material properties under various lighting conditions.

This tutorial will cover the following:

1. **BRDF Models Overview**: Understanding the BRDF models available in PyPBR.
2. **Evaluating BRDFs**: Using the Cook-Torrance model to evaluate material appearance.

By the end of this tutorial, you'll be comfortable with evaluating and rendering materials using the BRDF models provided by PyPBR.

BRDF Models Overview
^^^^^^^^^^^^^^^^^^^^

The **PyPBR** library provides various BRDF models to represent how light interacts with surfaces. The primary model available is the `CookTorranceBRDF`, which is widely used in physically based rendering due to its accuracy in representing real-world materials.

Cook-Torrance BRDF
------------------

The `CookTorranceBRDF` model is used to calculate the reflection of light from a surface, taking into account the surface roughness, Fresnel effects, and geometric occlusion. This model provides a balance between realism and computational efficiency.

Evaluating BRDFs
----------------

The `CookTorranceBRDF` class in PyPBR allows you to evaluate the BRDF for a given material, view direction, and light source. 

The **PyPBR** library allows you to render materials under different types of light sources, such as point lights and directional lights. Below, we provide examples of both types of rendering.

Point Light Rendering
"""""""""""""""""""""

.. image:: /_static/figures/point.png
    :alt: Directional Lighting Representation
    :width: 100%
    :align: center

A point light simulates a light source that emits light in all directions from a single point, like a light bulb. 
The light intensity decreases with distance, which creates distinct highlights and shadows.

.. code-block:: python

    import torch
    from pypbr.models import CookTorranceBRDF
    from pypbr.material import BasecolorMetallicMaterial
    from PIL import Image

    # Load a material
    albedo_map = Image.open("path/to/albedo_map.png")
    material = BasecolorMetallicMaterial(albedo=albedo_map)

    # Define the BRDF model with point light
    brdf = CookTorranceBRDF(light_type="point")

    # Define view and light directions
    view_dir = torch.tensor([0.0, 0.0, 1.0])  # Viewing straight on
    light_dir = torch.tensor([0.1, 0.1, 1.0])  # Light coming from slightly top right
    light_intensity = torch.tensor([1.0, 1.0, 1.0])  # White light

    # Evaluate the BRDF to get the reflected color
    reflected_color = brdf(material, view_dir, light_dir, light_intensity)

    # Print the reflected color
    print(reflected_color)

Directional Light Rendering
"""""""""""""""""""""""""""

.. image:: /_static/figures/directional.png
    :alt: Directional Lighting Representation
    :width: 100%
    :align: center

A directional light simulates light that is coming from a very far distance, like the sun. 
The light rays are parallel, and the intensity is uniform across the entire scene.

.. code-block:: python

    import torch
    from pypbr.models import CookTorranceBRDF
    from pypbr.material import BasecolorMetallicMaterial
    from PIL import Image

    # Load a material
    albedo_map = Image.open("path/to/albedo_map.png")
    material = BasecolorMetallicMaterial(albedo=albedo_map)

    # Define the BRDF model with directional light
    brdf = CookTorranceBRDF(light_type="directional")

    # Define view and light directions
    view_dir = torch.tensor([0.0, 0.0, 1.0])  # Viewing straight on
    light_dir = torch.tensor([1.0, 1.0, 1.0])  # Light coming from top right
    light_intensity = torch.tensor([1.0, 1.0, 1.0])  # White light

    # Evaluate the BRDF to get the reflected color
    reflected_color = brdf(material, view_dir, light_dir, light_intensity)

    # Print the reflected color
    print(reflected_color)

In these examples, we use both point and directional lights to evaluate the reflection of a material. This allows us to see the difference in appearance when the type of light changes.

Summary
^^^^^^^

In this tutorial, we covered the basics of BRDF evaluation and rendering using **PyPBR**:

1. **BRDF Models Overview**: Introduction to the Cook-Torrance BRDF model.
2. **Evaluating BRDFs**: How to evaluate the BRDF for a given material and light setup.
