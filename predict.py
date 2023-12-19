#!/usr/bin/env python3

"""_summary_

"""

from utils.load_and_save_checkpoint import load_checkpoint
from utils.get_input_args import get_predict_input_args
from utils.preprocess_image import process_image
from utils.load_categories_dict import load_categories
from utils.model import predict, get_pretrained_model


def main():
    """It does the actual prediction"""

    # get the input args of prediction
    args = get_predict_input_args()

    # load the trained model from the checkpoint
    model_pre = get_pretrained_model(args.arch, pretrained=True)
    model = load_checkpoint(args.checkpoint_path, model_pre, args.arch)

    # load the categories dictionary
    cat_to_name = load_categories(args.cat_path)

    # get the sample image to predict

    image_path = args.image_path

    # process the image
    image = process_image(image_path)

    # predict the the image category

    probs, classes = predict(image, model, args.top_k, cat_to_name, args.gpu)

    # print the class of the image with the highest probability.

    print(f"The image is predicted to be {classes[0]} with a probability of {probs[0]}")


if __name__ == "__main__":
    main()
