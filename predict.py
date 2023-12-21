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
    get_device
)


def main():
    """It does the actual prediction"""

    # get the input args of prediction
    args = get_predict_input_args()

    # load the categories dictionary
    cat_to_name = load_categories(args.categories_path)

    # get the sample image to predict

    image_path = args.image_path

    # process the image
    image = process_image(image_path)

    # get the device
    device = get_device(args.gpu)

    # load the trained model from the checkpoint
    model_pre = get_pretrained_model(args.arch, pretrained=True)
    model = load_checkpoint(args.checkpoints_dir, model_pre, args.arch)

    # predict the image category
    try:
        probs, classes, flowers = predict(
            image, model, args.top_k, cat_to_name, device=device
        )

        # print the class of the image with the highest probability.

        print(
            f"The image is predicted to be {flowers[0]} of id"
            f"{classes[0]} with a probability of {probs[0]}"
        )

        # print the top k classes of the image with the highest probability.
        print("The top k classes and their flower names are:\n")
        for i in range(args.top_k):
            print(f"{flowers[i]} of id {classes[i]} with a probability of {probs[i]}")
            print("========================================================================")

    except (
            AssertionError,
            ResourceWarning,
            UnboundLocalError,
    ) as e:
        print(f"Error predicting the image {str(e)}")


if __name__ == "__main__":
    main()
