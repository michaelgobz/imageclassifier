#!/usr/bin/env python3

"""This file contains the  work flow of the training the models
the acceptable model architectures are 
  - resnet
  - vgg
it trains the deep learning model based on those supported architectures
it runs on the gpu by default but also cpu but not advised to do so


"""
import os
import re
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
)


def main():
    """
    This is the main function that runs the training of the model

    Returns: None

    """
    # checkpoint path pattern
    flag = False

    start_time = time()
    # get the arguments
    optional_args = get_train_input_args()

    # define the device
    device = get_device(optional_args.gpu)

    # get the transforms
    train_tf, test_tf = get_transforms()

    # load the data
    trainloader, validloader, train_dataset = get_data(optional_args.data_directory, train_tf, test_tf)

    # get the pre-defined model

    pre_trained_model = get_pretrained_model(optional_args.arch, pretrained=True)

    # define the classifier
    model = create_the_classifier(pre_trained_model, optional_args.arch, optional_args.hidden_units)

    # define the loss function (criterion)

    criterion = define_loss_criterion()

    # define the optimizer

    optimizer = define_optimizer(model, optional_args.arch, optional_args.learning_rate)

    # check if checkpoint directory had checkpoints
    for file in os.listdir(optional_args.save_dir):
        # if the directory contains .pth files
        if re.match(r".*\.pth", file):
            flag = True
            break

    # train the model
    if flag and not optional_args.force:
        # load the checkpoint
        print("You have a checkpoint saved in the checkpoint directory\n")
        print(
            "Use the checkpoint you saved from previous training loop to continue \
                to the prediction stage\n"
        )
        print("load the checkpoint using the predict.sh script\n")
        print(
            "Still want to train the model from scratch? use the --force flag and run \
                the script again\n"
        )
        exit(1)

    elif flag and optional_args.force:
        try:
            print("Training the model from scratch")
            trained_model, optimizer = train(
                model,
                optional_args.epochs,
                trainloader,
                validloader,
                criterion,
                optimizer,
                device,
            )
            # save the model check point
            print(f"creating a checkpoint at {datetime.datetime.now()}")

            save_checkpoint(
                optional_args.save_dir,
                trained_model,
                optimizer,
                optional_args.learning_rate,
                optional_args.epochs,
                train_dataset,
                optional_args.arch,
            )

            print("Checkpointing complete")
        except (RuntimeError, ResourceWarning) as e:
            print(f"Error training the model {e}")
    else:
        try:
            print("Training the model from scratch")
            trained_model, optimizer = train(
                model,
                optional_args.epochs,
                trainloader,
                validloader,
                criterion,
                optimizer,
                device,
            )
            # save the model check point
            print(f"creating a checkpoint at {datetime.datetime.now()}")

            save_checkpoint(
                optional_args.save_dir,
                trained_model,
                optimizer,
                optional_args.learning_rate,
                optional_args.epochs,
                train_dataset,
                optional_args.arch,
            )

            print("Checkpointing complete")

        except (RuntimeError, ResourceWarning, AssertionError) as e:
            print(f"Error training the model {e}")

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
