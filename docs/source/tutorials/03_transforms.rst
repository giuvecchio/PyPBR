.. _material_transforms:

Material Transforms
========================

In this tutorial, we will explore techniques for transforming Physically-Based Rendering (PBR) materials using both functional and class-based approaches in **PyPBR**. We will cover how to apply geometric transformations, convert color spaces, adjust normal maps, and use the `Compose` method to apply multiple transformations in sequence.

1. **Geometric Transformations**: Resize, crop, rotate, tile, flip, and roll texture maps.
2. **Color Space Conversions**: Convert between sRGB and linear color spaces.
3. **Normal Map Adjustments**: Invert and adjust normal map strength.
4. **Composing Transformations**: Apply multiple transformations efficiently using `Compose`.

By the end of this tutorial, you will be able to apply a series of transformations to PBR materials for various rendering or artistic needs.

Functional vs. Class-Based Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyPBR** provides two main approaches for transforming PBR materials:

- **Functional Transforms**: Standalone functions that apply transformations to materials.
- **Class-Based Transforms**: Object-oriented transforms that can be composed and reused.

Using Functional Transforms
---------------------------

Here’s how to apply functional transforms, such as resizing, cropping, rotating, and tiling.

.. code-block:: python

    from pypbr.material import BasecolorMetallicMaterial
    from pypbr.functional import resize, crop, rotate, tile, flip_horizontal, to_linear
    from PIL import Image

    # Load a material
    albedo_map = Image.open("path/to/albedo_map.png")
    material = BasecolorMetallicMaterial(albedo=albedo_map)

    # Resize the material texture maps
    material = resize(material, size=(512, 512))

    # Crop the texture maps
    material = crop(material, top=10, left=10, height=300, width=300)

    # Rotate the texture maps by 45 degrees
    material = rotate(material, angle=45, expand=True)

    # Tile the texture maps 2 times
    material = tile(material, num_tiles=2)

    # Flip the texture maps horizontally
    material = flip_horizontal(material)

    # Convert albedo to linear space
    material = to_linear(material)

Using Class-Based Transforms
----------------------------

Class-based transforms in **PyPBR** allow you to define transformations in an object-oriented manner. Each transformation is encapsulated in a class and can be applied independently, offering flexibility and reusability for your material manipulation workflows.

.. code-block:: python

    from pypbr.material import BasecolorMetallicMaterial
    from pypbr.transforms import Resize, Crop, Rotate, Tile, FlipHorizontal, ToLinear
    from PIL import Image

    # Load a material
    albedo_map = Image.open("path/to/albedo_map.png")
    material = BasecolorMetallicMaterial(albedo=albedo_map)

    # Apply each transformation step by step

    # Resize the material texture maps
    resize_transform = Resize(size=(512, 512))
    material = resize_transform(material)

    # Crop the texture maps
    crop_transform = Crop(top=10, left=10, height=300, width=300)
    material = crop_transform(material)

    # Rotate the texture maps by 45 degrees
    rotate_transform = Rotate(angle=45, expand=True)
    material = rotate_transform(material)

    # Tile the texture maps 2 times
    tile_transform = Tile(num_tiles=2)
    material = tile_transform(material)

    # Flip the texture maps horizontally
    flip_transform = FlipHorizontal()
    material = flip_transform(material)

    # Convert the albedo to linear color space
    to_linear_transform = ToLinear()
    material = to_linear_transform(material)

    # Print the transformed material details
    print(material)

In this approach, each transformation is applied sequentially by creating an instance of the corresponding transform class and then calling it with the material as input. This offers modularity, as you can easily replace or rearrange individual transformations as needed.

Applying Multiple Transformations with Compose
----------------------------------------------

The `Compose` function allows you to combine multiple transformations into a single pipeline. 
This is particularly useful when you need to apply the same series of transformations in a specific order.

Here’s how to apply several transformations together using `Compose`:

.. code-block:: python

    from pypbr.transforms import Resize, Rotate, FlipVertical, AdjustNormalStrength, ToLinear
    from pypbr.functional import to_srgb

    # Define a series of transformations
    transform = Compose([
        Resize(size=(512, 512)),
        Rotate(angle=90, expand=True),
        FlipVertical(),
        AdjustNormalStrength(strength_factor=1.2),
        ToLinear()
    ])

    # Apply to material
    material = transform(material)

    # Further apply functional transformations (if needed)
    material = to_srgb(material)

    # Print the material details
    print(material)

This pipeline first resizes the material, rotates it, flips it vertically, adjusts the normal map strength, and converts the material’s texture maps to linear space, followed by a conversion back to sRGB color space using the functional approach.

Color Space Conversions
^^^^^^^^^^^^^^^^^^^^^^^

As mentioned earlier, accurate color space representation is crucial in PBR workflows. The **PyPBR** library allows conversion between linear and sRGB color spaces for the albedo and specular maps.

.. code-block:: python

    from pypbr.functional import to_linear, to_srgb

    # Convert albedo to linear space
    material = to_linear(material)

    # Convert albedo back to sRGB space
    material = to_srgb(material)

Normal Map Adjustments
^^^^^^^^^^^^^^^^^^^^^^

Adjusting normal maps allows for control over surface detail intensity. The library provides tools to invert the Y-axis of normal maps and adjust normal map strength.

.. code-block:: python

    from pypbr.functional import invert_normal_map, adjust_normal_strength

    # Invert the Y component of the normal map
    material = invert_normal_map(material)

    # Adjust the strength of the normal map by a factor of 1.5
    material = adjust_normal_strength(material, strength_factor=1.5)

    print(material)

Summary
^^^^^^^

In this tutorial, we explored how to manipulate PBR materials using both functional and class-based transformations in **PyPBR**. Key topics included:

1. **Geometric Transformations**: Resize, crop, rotate, tile, flip, and roll texture maps.
2. **Color Space Conversions**: Convert albedo and specular maps between sRGB and linear color spaces.
3. **Normal Map Adjustments**: Invert normal maps and adjust their strength.
4. **Compose**: Apply multiple transformations in sequence for efficient material processing.
