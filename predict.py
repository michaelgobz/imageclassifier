#!/usr/bin/env python3

"""File contains the call to the main function of the prediction
   author: Michael Goboola
   date: 2023-19-12
   time: 20:00

"""

from utils import (
    get_predict_input_args,
    load_checkpoint,
    load_categories,
    get_pretrained_model,
    process_image,
    predict,
)


def main():
    """It does the actual prediction"""

    # get the input args of prediction
    args = get_predict_input_args()

    # load the trained model from the checkpoint
    model_pre = get_pretrained_model(args.arch, pretrained=True)
    model = load_checkpoint(args.checkpoint_path, model_pre, args.arch)

    # load the categories dictionary
    cat_to_name = load_categories(args.categories_path)

    # get the sample image to predict

    image_path = args.image_path

    # process the image
    image = process_image(image_path)

    # predict the image category

    probs, classes, flowers = predict(image, model, args.top_k, cat_to_name, args.gpu)

    # print the class of the image with the highest probability.

    print(
        f"The image is predicted to be {flowers[0]} of id"
        f"{classes[0]} with a probability of {probs[0]}"
    )


if __name__ == "__main__":
    main()
