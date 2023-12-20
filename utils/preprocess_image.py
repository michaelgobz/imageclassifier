#!/usr/bin/env python3

"""file contains the function to preprocess the image for prediction
   author: Michael Goboola
    date: 2023-19-12
    time: 20:00
"""
import torch
import numpy as np
from PIL import Image


def process_image(image_path):

    """
    function to preprocess the image for prediction it does the following:
    1. resize the image to 256x256
    2. crop the image to 224x224
    3. convert the image to a numpy array
    4. normalize the image with the mean and std
    5. transpose the image to the correct format
    6. convert the image to a float tensor
    Args:
        image: path to the image

    Returns: the image as a  float tensor

    """
    with Image.open(image_path) as image:
        # resize the image
        image = image.resize((256, 256))
        # crop the image
        width, height = image.size
        new_width = 224
        new_height = 224
        left = int((width - new_width) / 2)
        top = int((height - new_height) / 2)
        right = int((width + new_width) / 2)
        bottom = int((height + new_height) / 2)
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
