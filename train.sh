#!/usr/bin/env bash

echo "Checking whether the data exists"

if [ ! -d "flowers" ]; then
    wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
    mkdir "./flowers"
    tar -xvzf flower_data.tar.gz -C ./flowers

else
    echo "flowers directory already exists"
    echo "proceeding to check whether the checkpoints directory exists"
fi

echo "--------------Checking whether the checkpoints directory exists-----------------"

if [ ! -d "checkpoints" ]; then
    echo "checkpoints directory does not exist"
    echo "creating checkpoints directory with name checkpoints"
    mkdir "./checkpoints"
else
    echo "checkpoints directory already exists"
    echo "proceeding to check the environment"
fi

echo "---------------Checking whether the environment exists-----------------------------"

if conda env list | grep -q  'base' ; then
    echo "base environment is already installed"
    conda activate base
else
    echo "base environment is not installed"\
    "creating base environment based on python 3.11.0 ..."
    conda create -n base python=3.11
    conda activate base
fi

echo "----------------Checking whether the required packages are installed----------------"

if conda list | grep -q 'matplotlib' ; then
    echo "matplotlib is already installed"
else
    echo "matplotlib is not installed"\
    "installing matplotlib ..."
    conda install -y matplotlib
fi

if conda list | grep -q 'numpy' ; then
    echo "numpy is already installed"
else
     echo "numpy is not installed"\
    "installing numpy ..."
     conda install -y numpy
fi

if conda list | grep -q 'torch' ; then
    echo "pytorch is already installed"
else
     echo "pytorch is not installed"\
    "installing pytorch ..."
     conda install -y pytorch torchvision -c pytorch

fi

# train the model
echo "Training has started"
echo "______________________________________________________________________________________________"

python train.py  ./flowers --gpu --epochs=1 --arch=resnet50 --learning_rate=0.001  --save_dir=./checkpoints --hidden_units=102

echo "______________________________________________________________________________________________"
echo "Training has finished"