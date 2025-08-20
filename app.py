import streamlit as st
from PIL import Image
import numpy as np

# Title of the app
st.title("Image Processing App")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using Pillow
    image = Image.open(uploaded_file)
    
    # Display the original image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert image to NumPy array for processing
    image_array = np.array(image)
    
    # Example processing: Convert to grayscale
    gray_image = Image.fromarray(np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8))
    
    # Display the processed image
    st.image(gray_image, caption='Processed Image (Grayscale)', use_column_width=True)
