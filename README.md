# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, I first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Part 1 - Developing an Image Classifier with Deep Learning

In this first part of the project, I work through a Jupyter notebook to implement an image classifier with PyTorch. I built and trained a deep neural network on the flower data set, which is a dataset of 102 flower categories.
the training is done on a GPU using google colab and GPU compute units provided by Udacity for training the model.
the model is trained using the resnet50 model over 10 optimization loops (epochs) and the accuracy is 71% on the test set.
the model is also trained using the vgg16 model over 10 optimization loops (epochs) and the accuracy is 78% on the test set.

* The notebooks are these:
  * [Image Classifier Project (resnet50).ipynb](Image%20Classifier%20Project%20(resnet50).ipynb)
  * [Image Classifier Project (vgg16).ipynb](Image%20Classifier%20Project%20(vgg16).ipynb)

## Part 2 - Building the command line application

After building and training a deep neural network on the flower data set, I converted it into an application that others can use. The application is a pair of Python scripts that run from the command line. For testing, I used the checkpoint I saved in the first part.

### Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.

#### Train a new network on a data set with train.py

* Basic usage: python train.py data_directory
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options:
  * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
  * Choose architecture: python train.py data_dir --arch "vgg16"
  * Set hyper-parameters: python train.py data_dir --learning_rate 0.01 --epochs 20
  * Use GPU for training: python train.py data_dir --gpu
  * set the number of hidden units for the output layer: python train.py data_dir --hidden_units default=102 because of project specifics but can be any depending on your specific classification problem.
  * The  --force flag trains the network even if a checkpoint is already saved in the save_directory.
  * Example: python train.py flowers --arch "vgg16" --learning_rate 0.01  --epochs 20 --gpu
  * The training loss, validation loss, and validation accuracy are printed out as a network trains and after training a checkpoint is saved in the save_directory.

#### Predict flower name from an image with predict.py along with the probability of that name. 

That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

* Basic usage: python predict.py /path/to/image checkpoint
* Options:
  * Return top K most likely classes: python predict.py input checkpoint --top_k 3
  * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
  * Use GPU for inference: python predict.py input checkpoint --gpu
  * Example: python predict.py ./flowers/test/1/image_06743.jpg ./checkpoints --top_k 5 --category_names cat_to_name.json --gpu
  * The top K classes along with associated probabilities are printed out after predicting.

#### Notes

* The model is trained on a GPU if available. which can be specified by adding the --gpu flag when running the program from the command line.
* The training loss, validation loss, and validation accuracy are printed out as a network trains and after training a checkpoint is saved in the save_directory.
* The model architectures supported are resnet50 and vgg16. the user can choose between them by using the --arch flag.
  * the default model is resent50 , but other models can be selected by using the --arch flag. the user can choose from resnet, vgg.
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

```md
image_06743.jpg

Top 5 classes: ['pink primrose', 'tree mallow', 'hibiscus', 'balloon flower', 'mexican petunia']
Top 5 probabilities: [0.9999, 0.0001, 0.0, 0.0, 0.0]
the correct class is: pink primrose

image_07090.jpg

[9.9436212e-01 3.3665085e-03 1.0850213e-03 1.0014515e-03 8.0418795e-05]
['10', '35', '54', '63', '50']
['globe thistle', 'alpine sea holly', 'sunflower', 'black-eyed susan', 'common dandelion']
the correct class is: globe thistle

image_07949.jpg

[0.70685905 0.13787131 0.08812183 0.04898518 0.00895899]
['96', '44', '58', '97', '59']
['camellia', 'poinsettia', 'geranium', 'mallow', 'orange dahlia']
the correct class is: camellia`

image_08004.jpg

[9.99982297e-01 1.09877265e-05 3.52990287e-06 1.25030022e-06 1.13775900e-06]
['102', '40', '90', '79', '6']
['blackberry lily', 'lenten rose', 'canna lily', 'toad lily', 'tiger lily']
the correct class is: blackberry lily

image_03098.jpg

[0.6033509  0.12982507 0.05468579 0.05096426 0.04762986]
['4', '68', '11', '40', '36']
['sweet pea', 'bearded iris', 'snapdragon', 'lenten rose', 'ruby-lipped cattleya']
the correct class is: sweet pea

image_03994.jpg

[9.9997449e-01 1.2721309e-05 6.3408888e-06 5.1764923e-06 4.8017961e-07]
['12', '100', '77', '50', '38']
["colt's foot", 'blanket flower', 'passion flower', 'common dandelion', 'great masterwort']
the correct class is: colt's foot

image_05769.jpg

[9.8769009e-01 1.1085894e-02 8.1221649e-04 1.8835271e-04 1.6258494e-04]
['13', '29', '17', '73', '77']
['king protea', 'artichoke', 'purple coneflower', 'water lily', 'passion flower']
the correct class is: king protea

image_06052.jpg

[0.8478797  0.10159896 0.03893943 0.00342476 0.00244155]
['14', '17', '29', '10', '67']
['spear thistle', 'purple coneflower', 'artichoke', 'globe thistle', 'spring crocus']
the correct class is: spear thistle

```

the  results are produced using resnet50 model

## Here is how to run the command line application

first clone the repository using the following command:

```bash
git clone https://github.com/michaelgobz/imageclassifier.git
```

then cd into the directory:

```bash
cd imageclassifier
```

then ger the data and create a checkpoints folder using for example the following command:
you change the folder names if you want to but remember to pass them to the train.py and predict.py files.

```bash
wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
mkdir ./checkpoints
mkdir ./flowers
```

then extract the data using the following command:

```bash
tar -xvzf flower_data.tar.gz -C ./flowers
```

then create a virtual environment using the following conda command:

```bash
conda create -n imageclassifier python=3.11.0
```

then activate the virtual environment using the following command:

```bash
conda activate imageclassifier
```

then run the train.py file using for example the following command:

```bash
python train.py ./flowers --gpu --epochs=10 --arch=resnet50 --learning_rate=0.001  --save_dir=./checkpoints
```

then run the predict.py file using for example the following command:

```bash
python predict.py ./flowers/test/1/image_06743.jpg ./checkpoints --arch=resnet50  --top_k=5 --categories_path=./cat_to_name.json --gpu
```

alternatively you can run the bash scripts using the following commands:
run the train.sh file first to check for train the model using the following command:

```bash
bash train.sh

```

then run the predict.sh file using the following command:

```bash
bash predict.sh
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

## conclusion

Thanks to Udacity for training provided in pytorch and the advanced mathematics. I enjoyed working on this project and learned a lot about deep learning and PyTorch.
I hope to use this knowledge in the future to build more advanced deep learning models.
Thanks to udacity and colab for providing the GPU for training the model.

## Author

[Michael Goboola](https://github.com/michaelgobz/)
