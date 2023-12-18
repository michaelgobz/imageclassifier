#!/usr/bin/env python3

"""_summary_

"""

from utils.load_and_save_checkpoint import load_checkpoint
from utils.get_input_args import get_predict_input_args


def main():
    """ It does the actual prediction """

    # get the input args of prediction
    args = get_predict_input_args()


    # load the trained model from the checkpoint
    model = load_checkpoint(args.checkpoint_path, args.arch)

    # get the sample image to predict

    # predict the the image category

    # print the class of the image with the highest probability.


if __name__ == "__main__":
    main()
