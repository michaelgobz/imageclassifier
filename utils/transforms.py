#!/usr/bin/env python3
"""
The data transforms to process the data 
make it ready for our machine learning model

author: Michael Goboola
date: 2023-19-12
time: 20:00
"""

from torchvision import transforms


def get_transforms():
    """function to get the transforms for the training and testing data
    it does the following:
    1. resize the image to 256
    2. crop the image to 224
    3. convert the image to a tensor
    4. normalize the image with the mean and std
    5. flip the image horizontally for the training data

    Returns:
       transform:  the transforms for the training and testing data
    """
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                          ])

    return train_transform, data_transform
