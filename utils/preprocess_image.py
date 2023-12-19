#!/usr/bin/env python3

"""_summary_
"""
import torch
import numpy as np
from PIL import Image


def process_image(image):
    
    """_Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array

    Returns:
        ndarray: numpy representation of the processed image 
    """
    image = Image.open(image)
    image = image.resize((256, 256))
    # crop the image
    width, height = image.size
    new_width = 224
    new_height = 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = image.crop((left, top, right, bottom))

    # convert to a numpy array
    image = np.array(image)
    image = image / 255
    # normalize the image with the mean and std
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    image = image.float()
    
    return image
