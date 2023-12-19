# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, I first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Part 1 - Developing an Image Classifier with Deep Learning

In this first part of the project, I work through a Jupyter notebook to implement an image classifier with PyTorch. I built and trained a deep neural network on the flower data set, which is a dataset of 102 flower categories.


## Part 2 - Building the command line application

After building and training a deep neural network on the flower data set, I converted it into an application that others can use. The application is a pair of Python scripts that run from the command line. For testing, I used the checkpoint I saved in the first part.


### Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.

#### Train a new network on a data set with train.py

* Basic usage: python train.py data_directory
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options:
  * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
  * Choose architecture: python train.py data_dir --arch "vgg13"
  * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --epochs 20
  * Use GPU for training: python train.py data_dir --gpu
  * Example: python train.py flowers --arch "vgg13" --learning_rate 0.01  --epochs 20 --gpu
  * The training loss, validation loss, and validation accuracy are printed out as a network trains and after training a checkpoint is saved in the save_directory.


#### Predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

* Basic usage: python predict.py /path/to/image checkpoint
* Options:
  * Return top K most likely classes: python predict.py input checkpoint --top_k 3
  * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
  * Use GPU for inference: python predict.py input checkpoint --gpu
  * Example: python predict.py flowers/test/1/image_06743.jpg checkpoint --top_k 5 --category_names cat_to_name.json --gpu
  * The top K classes along with associated probabilities are printed out after predicting.

#### Notes
* The model is trained on a GPU if available. which can be specified by adding the --gpu flag when running the program from the command line.
* The training loss, validation loss, and validation accuracy are printed out as a network trains and after training a checkpoint is saved in the save_directory.
* The training script allows users to choose from at least two different architectures available from torchvision.models.
   * the default model is resent , but other models can be selected by using the --arch flag. the user can choose from resnet, vgg.
   * The hyperparameters for learning rate, number of hidden units, and training epochs are set by the user.
   * The user can choose to train the model on a GPU by specifying the --gpu flag in the call to train.py. which is highly recommended.since its a deep neural network.


## Part 3 - Testing the command line application

To test the command line application, I downloaded the checkpoint I saved in the first part and used it to predict the top 5 most 
likely classes along with the probabilities on the following flower images:

* flowers/test/1/image_06743.jpg
* flowers/test/10/image_07090.jpg
* flowers/test/101/image_07949.jpg
* flowers/test/102/image_08004.jpg
* flowers/test/11/image_03098.jpg
* flowers/test/12/image_03994.jpg
* flowers/test/13/image_05769.jpg
* flowers/test/14/image_06052.jpg

The results were as follows:

```
image_06743.jpg


