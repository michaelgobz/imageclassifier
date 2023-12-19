#!/usr/bin/env bash

# check for the environment
# check if checkpoint exists in checkpoint directory

if [ ! -d "checkpoint/*.pth" ]; then
    echo "checkpoint directory does not exist  train the model first"
    exit 1

fi
# predict the flower name from an image with predict.py along with the probability of that name.


python predict.py ./flowers/test/1/image_06743.jpg --checkpoint=./checkpoint --top_k=5 --categories_path=./cat_to_name.json --gpu > predict.log.txt

