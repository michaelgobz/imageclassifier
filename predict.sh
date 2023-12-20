#!/usr/bin/env bash

# predict the flower name from an image with predict.py along with the probability of that name.

echo "Predicting has started"
echo "_____________________________________________________________________________________"

python predict.py ./flowers/test/1/image_06743.jpg ./checkpoints --arch=resnet50 --top_k=5 --categories_path=./cat_to_name.json --gpu

echo "_____________________________________________________________________________________"
echo "Predicting has finished"