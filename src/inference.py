import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from src.generator import Generator  # Import your model

# Load the trained model
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("./models/DeblurGAN_epoch_45.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
    return transform(image).unsqueeze(0)  # Add batch dimension: (1, 3, 256, 256)

# Deblur function
def deblur(image_path):
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output_tensor = model(image_tensor)  # Model output: (1, C, H, W)

    output_tensor = output_tensor.squeeze(0)  # Remove batch dimension: (C, H, W)

    # Ensure it has 3 channels (if single-channel, replicate across RGB channels)
    if output_tensor.shape[0] == 1:
        output_tensor = output_tensor.repeat(3, 1, 1)  # Convert grayscale to RGB

    # Clamp values to [0, 1] after reversing normalization to avoid invalid data types
    output_tensor = torch.clamp((output_tensor * 0.5) + 0.5, min=0.0, max=1.0)

    # Convert tensor to uint8 for compatibility with PIL
    tensor_to_pil = transforms.ToPILImage()
    output_image = tensor_to_pil((output_tensor * 255).to(torch.uint8))  

    return output_image

# Example usage
if __name__ == "__main__":
    input_image = "../datasets/test/sample_blurred.jpg"
    output_image = deblur(input_image)
    output_image.save("../datasets/test/sample_deblurred.jpg")
    print("Deblurred image saved successfully!")

