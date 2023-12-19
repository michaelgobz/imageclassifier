#!/usr/bin/env python3

"""function that obtains the command line arguments
   Author: Michael Goboola
   Date: 2023-19-12
   Time: 20:00

"""

import argparse


def get_train_input_args():
    """ get_train_input_args() function obtains the following command line arguments:
    thats used to train the model and save the model to a checkpoint and these are:
    1. data_dir - the directory of the training data
    2. arch - the model architecture
    3. learning_rate - the learning rate of the model
    4. save_dir - the directory to save the model to
    5. epochs - the number of times to train the model
    6. gpu - the option to use the GPU


    Returns:
        Namespace:  the parsed argument collection
    """

    parser = argparse.ArgumentParser()

    # add the expected arguments

    parser.add_argument(
        "data_directory",
        type=str,
        default="./flowers",
        help="directory containing the training data",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="resent50",
        help="The model architecture to be used in training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of times to train the model (optimization loop)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="The directory to save the model to checkpoints",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.003,
        help="The model leaning rate, small float percentage",
    )
    parser.add_argument(
        "--gpu", help="Tell the model to run on he GPU",
        action="store_true"

    )

    args = parser.parse_args()

    return args


def get_predict_input_args():
    """get_predict_input_args() function obtains the following command line arguments
    that are used to predict the class of an image and these are:

    1. path - the path to the image to be used for inference
    2. checkpoint - the path to the checkpoint to use for inference
    3. category_names - the category names of flowers
    4. top_k - get the top k predicted classes / categories
    5. gpu - the option to use the GPU
    6. arch - the model architecture


    Returns:
       Namespace : the parsed argument collection
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image to be used for inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/checkpoints/checkpoint_resnet50.pth",
        help="path to the checkpoint to use for inference",
    )

    parser.add_argument(
        "--categories_path",
        type=str,
        default="./cat_to_name.json",
        help="The category names of flowers",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Get the top k predicted classes / categories",
    )

    parser.add_argument(
        "--gpu",
        help="Tell the model to run on he GPU",
        action="store_true"
    )

    parser.add_argument("--arch",
                        type=str,
                        default="resent50",
                        help="model architecture")

    args = parser.parse_args()

    return args
