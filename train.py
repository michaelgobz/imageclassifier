#!/usr/bin/env python3

"""This file contains the the work flow of the training the models
the acceptable model architectures are 
  - resnet
  - vgg
it trains the deep learning model based on those supported architectures
it runs on the gpu by default but also cpu but not advised to do so


"""
import os
import datetime
from time import time

from utils import (
    get_device,
    get_pretrained_model,
    define_optimizer,
    define_loss_criterion,
    train,
    create_the_classifier,
    save_checkpoint,
    get_transforms,
    get_data,
    get_train_input_args,
    load_checkpoint,
)


def main(*args):
    """_summary_"""

    start_time = time()
    # get the arguments
    args = get_train_input_args()
    print(args)

    # define the device
    device = get_device(args="gpu")

    # get the transforms
    train_tf, test_tf = get_transforms()

    # load the data
    trainloader, validloader = get_data("/flowers", train_tf, test_tf)

    # get the pre-defined model

    pre_trained_model = get_pretrained_model(args.arch, pretrained=True)

    # define the classifier
    model = create_the_classifier(pre_trained_model, args.arch)

    # define the loss function (criterion)

    criterion = define_loss_criterion()

    # define the optimizer

    optimizer = define_optimizer(model, args.arch, args.learning_rate)

    # check if checkpoint directory had checkpoints
    if args.checkpoint:
        # if the directory contains .pth files
        if os.path.exists(args.save_dir + "*.pth"):
            # load the checkpoint
            model = load_checkpoint(args.checkpoint, model, args.arch)
    else:
        print("No checkpoint directory provided\n")
        print("provide the checkpoint directory to load or save the checkpoint")
        exit(1)
    
    # train the model

    trained_model, optimizer = train(
        model, args.epochs, trainloader, validloader, criterion, optimizer, device
    )

    # save the model check point
    print(f"creating a checkpoint at {datetime.datetime.now()}")

    save_checkpoint(
        args.save_dir,
        trained_model,
        optimizer,
        args.learning_rate,
        args.epochs,
        trainloader,
        args.arch,
    )

    print("Checkpointing complete")

    end_time = time()

    tot_time = end_time - start_time

    print("______________________________________________________________\n")
    print(
        f"The model training took {str(int((tot_time / 3600)))} :\
          {str(int((tot_time % 3600) / 60))} :\
          {str(int((tot_time % 3600) % 60))}"
    )


if __name__ == "__main__":
    main()
