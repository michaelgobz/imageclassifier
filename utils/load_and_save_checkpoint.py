#!/usr/bin/env python3

"""_summary_
"""
import torch
from torch import optim


def load_checkpoint(filepath, model_pretrained):
    """_summary_

    Args:
        filepath (_type_): _description_
        model_pretrained (_type_): _description_

    Returns:
        _type_: _description_
    """
    checkpoint = torch.load(filepath)

    optimizer = optim.Adam(
        model_pretrained.classifier.parameters(), lr=checkpoint["learning_rate"]
    )
    model_pretrained.classifier = checkpoint["classifier"]
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    model_pretrained.load_state_dict(checkpoint["state_dict"])

    return model_pretrained.classifier


def save_checkpoint(dirpath, model, optimizer, rate, epochs, train_data, arch):
    """_summary_

    Args:
        dirpath (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
        rate (_type_): _description_
        epochs (_type_): _description_
        train_data (_type_): _description_
        arch (_type_): _description_
    """
    model.class_to_idx = train_data.class_to_idx
    # the checkpoint
    checkpoint = {
        "epochs": epochs,
        "learning_rate": rate,
        "arch": arch,
        "classifier": model.classifier,
        "optimizer_state": optimizer.state_dict(),
        "class_to_idx": model.class_to_idx,
        "state_dict": model.state_dict(),
    }
    name = f"checkpoint_{arch}"

    torch.save(checkpoint, dirpath + "/" + name)
