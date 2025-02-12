.. _dataset_and_rendering_loss:

Advanced Use: Custom Dataset and Rendering Loss
===============================================

In this advanced tutorial, we will cover how to integrate the **PyPBR** library into a dataset class and use it for training a deep learning model with a **rendering loss**. Specifically, we'll look at how to:

1. **Create a Custom Dataset**: Use PyPBR to create a materials dataset class compatible with PyTorch.
2. **Rendering Loss Using BRDF**: Utilize the Cook-Torrance BRDF evaluation to compute a rendering loss.

By the end of this tutorial, you will understand how to leverage PyPBR's BRDF evaluation to create a more photorealistic loss for training deep learning models.

Creating a Custom Dataset with PyPBR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The **PyPBR** library can be easily integrated into a PyTorch dataset to facilitate the loading and handling of PBR materials. 

Below is a custom PyTorch dataset class that loads PBR materials from folders.

.. code-block:: python

    import os
    import torch
    from torch.utils.data import Dataset
    from pypbr.material import load_material_from_folder

    class PBRMaterialDataset(Dataset):
        def __init__(self, root_dir, color_space='srgb', workflow='metallic', resolution=(256, 256)):
            """
            Initialize the dataset by specifying the root directory containing material folders.

            Args:
                root_dir (str): The root directory containing the material folders.
                color_space (str): Color space for the albedo map, either 'srgb' or 'linear'. Defaults to 'srgb'.
                workflow (str): Preferred workflow, either 'metallic' or 'specular'. Defaults to 'metallic'.
                resolution (tuple): Desired resolution for the texture maps. Defaults to (256, 256).
            """
            self.root_dir = root_dir
            self.material_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            self.color_space = color_space
            self.workflow = workflow
            self.resolution = resolution

        def __len__(self):
            return len(self.material_dirs)

        def __getitem__(self, idx):
            folder_path = self.material_dirs[idx]

            # Load the material from the folder
            is_srgb = self.color_space == 'srgb'
            material = load_material_from_folder(folder_path, preferred_workflow=self.workflow, is_srgb=is_srgb)
            
            # Resize the material to the desired resolution
            material.resize(self.resolution)

            return material

The `PBRMaterialDataset` allows for easy loading of materials from a directory. You can now use this dataset with standard PyTorch data loaders to provide batches of PBR materials during training.

Rendering Loss Using BRDF
^^^^^^^^^^^^^^^^^^^^^^^^^

To use PyPBR for a rendering loss, we can evaluate the BRDF of the predicted material properties and compare it with a reference rendering. This type of loss encourages the network to learn physically plausible material properties.

In this tutorial, we will use the `CookTorranceBRDF` model to evaluate both a predicted material and a ground truth material, and compute a rendering loss based on the difference between the two.

.. code-block:: python

    import torch
    import torch.nn as nn
    from pypbr.models import CookTorranceBRDF

    class RenderingLoss(nn.Module):
        def __init__(self, light_type='point', view_dir=torch.tensor([0.0, 0.0, 1.0]), light_dir=torch.tensor([0.1, 0.1, 1.0]), light_intensity=torch.tensor([1.0, 1.0, 1.0])):
            """
            Initialize the rendering loss module.

            Args:
                light_type (str): The type of light ('point' or 'directional'). Defaults to 'point'.
                view_dir (torch.Tensor): The view direction vector. Defaults to viewing straight on.
                light_dir (torch.Tensor): The light direction vector. Defaults to light from slightly top right.
                light_intensity (torch.Tensor): The light intensity. Defaults to white light.
            """
            super(RenderingLoss, self).__init__()
            self.brdf = CookTorranceBRDF(light_type=light_type)
            self.view_dir = view_dir
            self.light_dir = light_dir
            self.light_intensity = light_intensity

        def forward(self, predicted_material, ground_truth_material):
            """
            Forward pass to compute the rendering loss.

            Args:
                predicted_material: Predicted material properties.
                ground_truth_material: Ground truth material properties.

            Returns:
                torch.Tensor: The computed rendering loss.
            """
            # Compute rendered color for predicted and ground truth materials
            rendered_pred = self.brdf(predicted_material, self.view_dir, self.light_dir, self.light_intensity)
            rendered_gt = self.brdf(ground_truth_material, self.view_dir, self.light_dir, self.light_intensity)

            # Compute L2 loss between the rendered outputs
            loss = nn.MSELoss()(rendered_pred, rendered_gt)
            return loss

Usage Example
-------------

Below is an example of how to use the `PBRMaterialDataset` and `RenderingLoss` in a typical training loop.

.. code-block:: python

    from torch.utils.data import DataLoader

    # Initialize dataset and dataloader
    root_dir = "path/to/materials_root"
    dataset = PBRMaterialDataset(root_dir, color_space='srgb', workflow='metallic', resolution=(256, 256))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize the rendering loss
    rendering_loss = RenderingLoss(light_type='point')

    # Example training loop
    for batch in dataloader:
        predicted_material = batch  # Here, replace this with your model's predicted material output
        ground_truth_material = batch  # Assuming the batch itself is the ground truth material for this example

        # Compute rendering loss
        loss = rendering_loss(predicted_material, ground_truth_material)

        # Print the loss
        print("Rendering Loss:", loss.item())

        # Perform backpropagation and optimization steps here
        ...

Summary
^^^^^^^

In this tutorial, we covered:

1. **Dataset Integration**: How to load materials into a PyTorch dataset using **PyPBR**.
2. **Rendering Loss**: How to create a custom rendering loss using BRDF evaluation.

By leveraging **PyPBR**, you can incorporate physically-based rendering into your deep learning workflows, helping models learn realistic material properties and improving the visual quality of results.
