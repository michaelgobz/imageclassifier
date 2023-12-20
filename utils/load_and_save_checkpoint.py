#!/usr/bin/env python3

"""File contains the functions to load and save the checkpoint of the trained model
   author: Michael Goboola
    date: 2023-19-12
"""
import torch
from torch import optim


def load_checkpoint(file_dir, model_pretrained, arch):
    """loads the saved checkpoint from a file path

    Args:
        file_dir (_str_): directory path to the checkpoints
        model_pretrained (_torch.Module_): pytorch pretrained model
        arch (_str_): architecture  of the model

    Returns:
        torch.Module: restored model from the checkpoint
    """
    try:
        checkpoint = torch.load(file_dir + "/checkpoint_" + arch + ".pth")
        model = None
        
        if arch == "resent50":
            optimizer = optim.Adam(
                model_pretrained.fc.parameters(), lr=checkpoint["learning_rate"]
            )
            model_pretrained.fc = checkpoint["classifier"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            model_pretrained.load_state_dict(checkpoint["state_dict"])
            model_pretrained.fc.class_to_idx = checkpoint["class_to_idx"]
            model = model_pretrained
        elif arch == "vgg16":
            optimizer = optim.Adam(
                model_pretrained.classifier.parameters(), lr=checkpoint["learning_rate"]
            )
            model_pretrained.classifier = checkpoint["classifier"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            model_pretrained.load_state_dict(checkpoint["state_dict"])
            model_pretrained.class_to_idx = checkpoint["class_to_idx"]
            model = model_pretrained
            
        return model

    except FileNotFoundError as e:
        print(f"Error loading checkpoint {e}")


def save_checkpoint(dirpath, model, optimizer, rate, epochs, train_data, arch):
    """saves the checkpoint of a trained model to a file path

    Args:
        dirpath (_str_): directory path to save the checkpoint
        model (_torch.MODULE_): trained model
        optimizer (_torch.Optim_): optimizer
        rate (_float_): learning rate
        epochs (_int_): Number of epochs
        train_data (_dataloader_): training data from the dataset
        arch (_str_):  model architecture
    """
    try:
        if arch == "resnet50":
            model.fc.class_to_idx = train_data.class_to_idx
            # the checkpoint
            checkpoint = {
                "epochs": epochs,
                "learning_rate": rate,
                "arch": arch,
                "classifier": model.fc,
                "optimizer_state": optimizer.state_dict(),
                "class_to_idx": model.fc.class_to_idx,
                "state_dict": model.state_dict(),
            }

            name = f"checkpoint_{arch}"
            torch.save(checkpoint, dirpath + "/" + name + ".pth")
            
        elif arch == "vgg16":
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
            torch.save(checkpoint, dirpath + "/" + name + ".pth")

    except FileNotFoundError as e:
        print(f"Error saving checkpoint {e}")
