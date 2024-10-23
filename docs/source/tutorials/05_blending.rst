.. _materials_blending:

Materials Blending
==================

In this tutorial, we will cover how to blend Physically Based Rendering (PBR) materials using the **PyPBR** library. Blending allows you to create complex materials by combining existing ones, either using masks, height maps, or other blending methods.

This tutorial will cover the following:

1. **Blending using the Functional API**: Blend materials using the functional approach for quick and direct usage.
2. **Blending using the Class-based API**: Blend materials using object-oriented blending classes for more structured and reusable workflows.
3. **Different Blending Methods**: Showcase the different blending methods such as mask blending, height-based blending, property blending, and gradient blending.

By the end of this tutorial, you will understand how to blend PBR materials using **PyPBR** effectively.

Blending using the Functional API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The functional API of **PyPBR** provides straightforward functions to blend materials. This is useful for quick and direct blending without the need for creating blending objects.

Mask Blending
-------------

You can blend two materials using a mask, where the mask determines how much of each material is used in the final blend.

.. code-block:: python

    from pypbr.material import BasecolorMetallicMaterial
    from pypbr.blending.functional import blend_with_mask
    from PIL import Image
    import torch

    # Load materials
    albedo_map1 = Image.open("path/to/albedo_map1.png")
    albedo_map2 = Image.open("path/to/albedo_map2.png")
    material1 = BasecolorMetallicMaterial(albedo=albedo_map1)
    material2 = BasecolorMetallicMaterial(albedo=albedo_map2)

    # Create a blending mask
    mask = torch.ones((1, 256, 256)) * 0.5  # A mask with 50% blend

    # Blend materials using the mask
    blended_material = blend_with_mask(material1, material2, mask)

    # Print blended material details
    print(blended_material)

Height-based Blending
---------------------

Blend two materials based on their height maps to achieve a more natural blending effect, particularly useful for blending terrain or other textured surfaces.

.. code-block:: python

    from pypbr.blending.functional import blend_on_height

    # Blend materials based on height maps
    blended_material = blend_on_height(material1, material2, blend_width=0.1)

    # Print blended material details
    print(blended_material)

Blending using the Class-based API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The class-based API in **PyPBR** provides an object-oriented approach to blending. This is ideal for more complex blending tasks where reusability and structure are important.

Mask Blend using Class-based API
--------------------------------

.. code-block:: python

    from pypbr.blending.blending import MaskBlend

    # Create a MaskBlend instance
    mask_blend = MaskBlend(mask=mask)

    # Apply the blending method to the materials
    blended_material = mask_blend(material1, material2)

    # Print blended material details
    print(blended_material)

Height-based Blend using Class-based API
----------------------------------------

.. code-block:: python

    from pypbr.blending.blending import HeightBlend

    # Create a HeightBlend instance
    height_blend = HeightBlend(blend_width=0.1)

    # Apply the blending method to the materials
    blended_material = height_blend(material1, material2)

    # Print blended material details
    print(blended_material)

Different Blending Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

Property-based Blending
-----------------------

Blend two materials based on a specific property, such as metallic or roughness. This allows for more nuanced control over how the materials are blended.

.. code-block:: python

    from pypbr.blending.functional import blend_on_properties

    # Blend materials based on the 'metallic' property
    blended_material = blend_on_properties(material1, material2, property_name="metallic", blend_width=0.1)

    # Print blended material details
    print(blended_material)

Gradient Blending
-----------------

You can blend two materials using a linear gradient. This is useful for smoothly transitioning between two materials.

.. code-block:: python

    from pypbr.blending.functional import blend_with_gradient

    # Blend materials using a horizontal gradient
    blended_material = blend_with_gradient(material1, material2, direction="horizontal")

    # Print blended material details
    print(blended_material)

Summary
^^^^^^^

In this tutorial, we explored different ways to blend materials using **PyPBR**:

1. **Functional API**: A quick and direct way to blend materials.
2. **Class-based API**: A structured approach to blending for complex scenarios.
3. **Blending Methods**: Different blending methods including mask blending, height-based blending, property blending, and gradient blending.