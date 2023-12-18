#!/usr/bin/env python3

"""_summary_
"""
import torch
from torch import optim, nn
from torchvision import models

from utils.preprocess_image import process_image
from utils.load_categories_dict import load_categories

def get_pretrained_model(arch: str, pretrained:bool=True):
    """_summary_

    Args:
        arch (str): _description_
        pretrained (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    pretrained_model = None
    if arch == 'resent' and pretrained:
        pretrained_model = models.resnet50(pretrained)
        
    elif pretrained == 0 and arch == 'resent':
        pretrained_model = models.resnet50(weights='IMAGENET_1k')
    
    elif pretrained and arch ='vgg':
        pretrained_model = models.vgg16(pretrained)
        
    elif pretrained = 0 and arch='vgg':
        pretrained_model  = models.vgg16(weights='IMAGENET_1k')
   
    else:
        print(f'model arch and pretrained bool need to be provided, supported archs vgg and resnet) and pretrained either true or false')
        print(f're-run the training with the correct parameters {arch , pretrained}')
        exit(1)
        
    return pretrained_model


def create_the_classifier(model, arch):
    
    """_summary_

    Returns:
        _type_: _description_
    """
    
    if arch == 'resent':
        model.fc = nn.Sequential(nn.Linear(2048, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1)
                                 )
    elif arch == 'vgg':
        model.classifier = nn.Sequential(nn.Linear(25088, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1)
                                 )
    
    return model

def define_loss_criterion():
    """_summary_

    Returns:
        _type_: _description_
    """
    
    criterion = nn.NLLLoss()
    return criterion
    

def define_optimizer(model, arch, learning_rate):
    
    """_summary_

    Returns:
        _type_: _description_
    """
    
    if arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    elif arch == 'vgg':
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return optimizer


def get_device(args=None):
    """_summary_

    Args:
        args (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    device = None
    if args is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cuda'
        
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
    print_every=10):

    # losses
    running_loss = 0
    valid_loss = 0

    # Accuracies
    valid_accuracy = 0
    accuracies = []

    #losses
    validationlosses = []
    traininglosses = []
    
    # move the model to device
    model.to(device)
    
    # run the training loop
    for epoch in range(epochs):
        print(f'Training Epoch {epoch} ...')
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
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                traininglosses.append(running_loss/print_every)
                validationlosses.append(valid_loss/print_every)
                accuracies.append((valid_accuracy/len(validloader))*100)
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/print_every:.3f}.. "
                    f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")
                running_loss = 0
                valid_loss = 0
                valid_accuracy = 0
                model.train()
    
    return model


# predict using the model

def predict(image_path, model, device, cat_path, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to(device)
    image = process_image(image_path)
    image = image.unsqueeze(0)
    print(image.shape)
    image = image.to(device)
    
    # categories
    cat_to_name:dict = load_categories(cat_path)
    
    with torch.no_grad():
        logps = model.forward(image)
        ps = logps
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[lab] for lab in top_class]
        top_flowers = [cat_to_name[lab] for lab in top_class]
        
    return top_p, top_class, top_flowers
    
    