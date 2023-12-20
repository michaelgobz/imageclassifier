#!/usr/bin/env bash

# predict the flower name from an image with predict.py along with the probability of that name.

echo "Predicting has started"

python predict.py ./flowers/test/1/image_06743.jpg --arch=vgg --checkpoints_dir=./checkpoints --top_k=5 --categories_path=./cat_to_name.json --gpu

echo "Predicting has finished"