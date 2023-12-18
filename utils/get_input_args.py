#!/usr/bin/env python3

"""function that obtains the command line arguments
   
   Author: Michael Goboola
"""

import argparse


def get_train_input_args():
    """_summary_

    Returns:
        _type_: _description_
    """

    parser = argparse.ArgumentParser

    # add the expected arguments

    parser.add_argument(
        "--dir",
        type == str,
        default="/flowers",
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
        "--save-dir",
        type=str,
        default="/checkpoints",
        help="The directory to save the model to checkpoints",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.003,
        help="The model leaning rate, small float percentage",
    )
    parser.add_argument(
        "--gpu", type=str, default="", help="Tell the model to run on he GPU"
    )

    args = parser.parse_args()

    return args


def get_predict_input_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/images/image.jpg",
        help="Path to the image to be used for inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/checkpoints/checkpoint_resnet50.pth",
        help="path to the checkpoint to use for inference",
    )

    parser.add_argument(
        "--category_names",
        type=str,
        default="/cat_to_name.json",
        help="The category names of flowers",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Get the top k predicted classes / categories",
    )

    parser.add_argument(
        "--gpu", type=str, default="", help="Tell the model to run on he GPU"
    )

    parser.add_argument("--arch", type=str, default="resent", help="model architecture")

    args = parser.parse_args()

    return args
