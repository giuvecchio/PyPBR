import torch
from torchvision.transforms.functional import to_pil_image

from pypbr.models import CookTorranceBRDF
from pypbr.io import load_material_from_folder

# Load material
material = load_material_from_folder("./data/tiles", preferred_workflow="metallic")

H, W = 512, 512
material.resize((H, W)).tile(2)

# Create an instance of the BRDF with the material
brdf = CookTorranceBRDF(light_type="point")

# Define the view direction, light direction, and light intensity
view_dir = torch.tensor([0.0, 0.0, 1.0])  # Viewing straight on
light_dir = torch.tensor([0.1, 0.1, 1.0])  # Light coming from slightly top right
light_intensity = torch.tensor([1.0, 1.0, 1.0])  # White light
light_size = 1.0

# Evaluate the BRDF to get the reflected color
reflected_color = brdf(material, view_dir, light_dir, light_intensity, light_size)

# Convert to PIL Image and display
image = to_pil_image(reflected_color)
image.show()
