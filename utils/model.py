#!/usr/bin/env python3

"""file contains all the functions to train the model
 author: Michael Goboola
    date: 2023-19-12
    time: 20:00

"""
import torch
from torch import optim, nn
from torchvision import models

from utils.preprocess_image import process_image
from utils.load_categories_dict import load_categories


def get_pretrained_model(arch: str, pretrained: bool = True):
    """function to get the pretrained model based on the provided architecture

    Args:
        arch (str) model architecture  which is required to build the model
        pretrained (bool, optional): tells torch to get the pretrained version. Defaults to True.

    Returns:
        _torch.Module_: a pretrained model
    """
    pretrained_model = None
    if arch == "resent50" and pretrained:
        pretrained_model = models.resnet50(pretrained=pretrained)

    elif pretrained == 0 and arch == "resent50":
        pretrained_model = models.resnet50(weights="IMAGENET_1k")

    elif pretrained and arch == "vgg16":
        pretrained_model = models.vgg16(pretrained=pretrained)

    elif pretrained == 0 and arch == "vgg16":
        pretrained_model = models.vgg16(weights="IMAGENET_1k")

    else:
        print(
            "model arch and pretrained bool need to be provided, supported architectures are vgg16 and resnet50)\
            and pretrained either true or false"
        )
        print(f"re-run the training with the correct parameter --arch to vgg16 or resent50 yours is {arch}")
        exit(1)

    return pretrained_model


def create_the_classifier(model, arch):
    """ This creates the classifier on top of the pretrained model so thats we use transfer learning
    to train the model on our data  with is a multi-class classification
    Returns:
        _torch.Module_: the model with the classifier bit changed to have out features of 102 and inputs depending on
        the model architecture
    """

    if arch == "resnet50":
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 102),
            nn.LogSoftmax(dim=1),
        )
    elif arch == "vgg16":
        model.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 102),
            nn.LogSoftmax(dim=1),
        )

    return model


def define_loss_criterion():
    """ creates the loss criterion

    Returns:
        _NLLLoss_: the loss criterion for a multi-class classification
    """

    criterion = nn.NLLLoss()
    return criterion


def define_optimizer(model, arch, learning_rate):
    """create the optimizer for the model based on the architecture

    Returns:
        optimizer : the optimizer for the model based on the architecture
    """

    if arch == "resnet50":
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    elif arch == "vgg16":
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return optimizer


def get_device(gpu=False):
    """get the device to run the model on

    Args:
        gpu: boolean to tell the model to run on the gpu or not

    Returns:
        _str_: device string representing the device to run the model on
    """

    device = None
    if gpu:
        device = "cuda"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def train(
        model,
        epochs,
        trainloader,
        validloader,
        criterion,
        optimizer,
        device,
        steps_for_eval=0,
        print_every=10,
):
    """ The function to train the model that trains the model over the optimisation loop and prints the training
    and validation loss and accuracy.

    Args:
        model (_torch.MODULE_): the model to train
        epochs (_int_): number of epochs to train the model
        trainloader (_dataloader_): loader for the training data
        validloader (_dataloader_): loader for the validation data
        criterion (_NLLLoss_): criterion to use for the loss
        optimizer (_optimizer_): optimizer to use for the model
        device (_str_): _description_
        steps_for_eval (int, optional): step to evaluate the model during training. Defaults to 0.
        print_every (int, optional): print the evaluation metrics per these steps. Defaults to 10.

    Returns:
        model, optimizer: returns the trained model and the optimizer
    """
    # losses
    running_loss = 0
    valid_loss = 0

    # Accuracies
    valid_accuracy = 0
    accuracies = []

    # losses
    validationlosses = []
    traininglosses = []

    # move the model to device
    model.to(device)

    # run the training loop
    for epoch in range(epochs):
        print(f"Training Epoch {epoch} ...")
        for inputs, labels in trainloader:
            steps_for_eval += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps_for_eval % print_every == 0:
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(
                            equals.type(torch.FloatTensor)
                        ).item()

                traininglosses.append(running_loss / print_every)
                validationlosses.append(valid_loss / print_every)
                accuracies.append((valid_accuracy / len(validloader)) * 100)

                print(
                    f"Epoch {epoch + 1}/{epochs}.. "
                    f"Train loss: {running_loss / print_every:.3f}.. "
                    f"Validation loss: {valid_loss / print_every:.3f}.. "
                    f"Validation accuracy: {valid_accuracy / len(validloader):.3f}"
                )
                running_loss = 0
                valid_loss = 0
                valid_accuracy = 0
                model.train()

    return model, optimizer


# predict using the model


def predict(image_path, model, device, cat_path, top_k=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    Args:
        image_path: path to the image to be predicted
        model:  the trained model to use for prediction
        device: the device to run the model on
        cat_path: the path to the categories json file
        top_k: the top k classes predicted for the image

    Returns: top_p, top_class, top_flowers the probabilities, classes and flowers predicted for the image

    """
    model.eval()
    model.to(device)
    image = process_image(image_path)
    image = image.unsqueeze(0)
    print(image.shape)
    image = image.to(device)

    # categories
    cat_to_name: dict = load_categories(cat_path)

    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(top_k, dim=1)
        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[lab] for lab in top_class]
        top_flowers = [cat_to_name[lab] for lab in top_class]

    return top_p, top_class, top_flowers
