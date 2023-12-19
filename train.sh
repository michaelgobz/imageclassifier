#!/usr/bin/env bash

if [ ! -d "flowers" ]; then
    wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
    mkdir "./flowers"
    unzip flower_data.zip -d flowers
fi

if [ ! -d "checkpoint" ]; then
    mkdir "./checkpoint"
fi

# check for the environment

if conda env list | grep -q  'base' ; then
    echo "base environment is already installed"
    conda activate base
else
    conda create -n base python=3.11
    conda activate base
fi

# check for the dependencies and install the missing ones

if conda list | grep -q 'matplotlib' ; then
    echo "matplotlib is already installed"
else
    conda install -y matplotlib
fi

if conda list | grep -q 'numpy' ; then
    echo "numpy is already installed"
else
    conda install -y numpy
fi

if conda list | grep -q 'torch' ; then
    echo "pytorch is already installed"
else
    conda install -y pytorch torchvision -c pytorch

fi

# train the model

python train.py  ./flowers --gpu --epochs=10 --arch=vgg16 --learning_rate=0.001  --save_dir=./checkpoint >