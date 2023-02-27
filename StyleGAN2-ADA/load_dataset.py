# Import necessary libraries
import tensorflow as tf
import os
import numpy as np
import PIL.Image
import zipfile
import requests

# Set the path to the dataset
data_dir = 'data'

# Download the CelebA dataset
if not os.path.exists(data_dir):
    print('Downloading CelebA dataset...')
    os.makedirs(data_dir)
    url = 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    file_name = os.path.join(data_dir, 'img_align_celeba.zip')
    with open(file_name, "wb") as file:
        response = requests.get(url)
        file.write(response.content)
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(file_name)

# Preprocess the dataset
def preprocess_image(image):
    # Crop the image to remove background and keep only the face
    image = image.crop((25, 50, 175, 200))
    # Resize the image to a uniform size
    image = image.resize((128, 128), resample=PIL.Image.LANCZOS)
    # Normalize the pixel values to [-1, 1]
    image = (np.asarray(image, dtype=np.float32) / 255.0) * 2 - 1
    return image

# Load the dataset
def load_dataset(data_dir):
    # Get the list of image files in the dataset directory
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    # Load and preprocess the images
    images = []
    for file in image_files:
        image = PIL.Image.open(file)
        image = preprocess_image(image)
        images.append(image)
    # Convert the list of images to a NumPy array
    images = np.array(images, dtype=np.float32)
    return images

# Load the CelebA dataset
dataset = load_dataset(os.path.join(data_dir, 'img_align_celeba'))
