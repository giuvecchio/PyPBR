.. PyPBR documentation master file, created by
   sphinx-quickstart on Wed Oct  9 16:06:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/giuvecchio/PyPBR

PyPBR documentation
===================

.. image:: /_static/logo.png
    :alt: Rendered Material with Directional Light
    :width: 100px
    :align: center

**PyPBR** is a Python library for creating, manipulating, and rendering Physically Based Rendering (PBR) materials, with a focus on providing a modular, flexible approach to material creation and blending. 

PyPBR supports multiple PBR workflows, such as basecolor-metallic and diffuse-specular, and includes functionality for BRDF evaluation and blending operations.

PyPBR is build on **PyTorch** to leverage GPU acceleration and for easy integration with existing AI codebases. 

Overview
--------

PyPBR aims to provide a user-friendly yet powerful toolkit for working with PBR materials in Python. It includes functionalities for:

- Creating different types of PBR materials.
- Manipulating texture maps (resize, rotate, crop, etc.).
- Evaluating Bidirectional Reflectance Distribution Functions (BRDF) using models like Cook-Torrance.
- Blending materials using a variety of methods (masks, height maps, gradients).
- Loading and saving materials from/to folders.


Installation
------------

You can install **PyPBR** via pip:

.. code-block:: sh

   pip install pypbr

For more detailed installation instructions, visit the :ref:`getting-started` page.

What is PBR?
------------

Physically Based Rendering (PBR) is an approach in computer graphics that aims to render materials in a way that mimics their appearance in the real world. PBR is widely used in the graphics and visual effects industries due to its ability to produce realistic and consistent results under various lighting conditions. PyPBR helps facilitate these workflows with easy-to-use Python tools.

Contribution
------------

Contributions are highly encouraged! Whether it's a bug report, a feature request, or a pull request, your involvement is valuable to us. Head over to the :ref:`contributing` page for more information.


Contact
-------

For questions, feature suggestions, or bug reports, feel free to open an issue on our GitHub repository. 
We value community feedback and collaboration!

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   contributing
   api/index
   tutorials/index