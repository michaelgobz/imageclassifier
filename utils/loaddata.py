#!/usr/bin/env python3
"""File contains the function to load the data from the data directory
   author: Michael Goboola
    date: 2023-19-12
    time 20:00
"""
import torch
from torchvision import datasets


def get_data(data_dir, train_transform, test_transform, batch_size=64):
    """
    function to load the data from the data directory
    Args:
        data_dir: data directory
        train_transform: torchvision transform for training data
        test_transform:  torchvision transform for testing data
        batch_size: batch size for the data loader

    Returns: the train and valid data loaders

    """

    train = datasets.ImageFolder(data_dir + "/train", transform=train_transform)
    valid = datasets.ImageFolder(data_dir + "/valid", transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)

    return trainloader, validloader, train
