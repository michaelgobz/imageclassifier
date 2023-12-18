#!/usr/bin/env python3

"""_summary_
"""

from torch import optim, nn
from torchvision import models


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
