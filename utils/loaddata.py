#!/usr/bin/env python3
"""_summary_
"""
import torch
from torchvision import datasets


def get_data(data_dir, train_transform, test_transform, batch_size=64):
    """

    Args:
        dir (str): _description_
    """

    train = datasets.ImageFolder(data_dir + "/train", transform=train_transform)
    valid = datasets.ImageFolder(data_dir + "/valid", transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)

    return trainloader, validloader
