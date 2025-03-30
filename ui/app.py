import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generator import Generator  # Import Generator model
from src.inference import deblur  # Import deblurring function

# Load the trained model
model = Generator()
model.load_state_dict(torch.load("./models/DeblurGAN_epoch_45.pth", map_location="cpu"))
model.eval()

st.title("DeblurGAN - Image Deblurring")

# Drag and drop file uploader
uploaded_file = st.file_uploader("Upload a blurred image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display images side by side with reduced size
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((256, 256))  # Resize image
        st.image(image, caption="Blurred Image", width=256)
    
    with col2:
        # Deblur the image
        deblurred_image = deblur(uploaded_file)
        deblurred_image = deblurred_image.resize((256, 256))  # Resize output
        st.image(deblurred_image, caption="Deblurred Output", width=256)
