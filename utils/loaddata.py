#!/usr/bin/env python3
"""_summary_
"""
import torch
from torchvision import datasets

from utlis.transforms import data_transform, train_transform


def get_data(dir: str, train_transform=train, test_transform=test, batch_size=64):
    """

    Args:
        dir (str): _description_
    """

    train = datasets.ImageFolder(dir + "/train", transform=train_transform)
    valid = datasets.ImageFolder(dir + "/valid", transform=test_transform)
    test = datasets.ImageFolder(dir + "/test", transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return trainloader, validloader, testloader
