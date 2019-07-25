# U-Net: Convolutional Network for Biomedical Image Segmentation
Implemantation of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) architecture using [Keras](https://keras.io/) framework

## Brief info

U-Net is a convolutional network architecture for fast and precise segmentation of images. 
Input is a grey scale 512x512 image in jpeg format, output - a 512x512 mask in png format. 
Below is the implemented model's architecture
<img src="https://pp.userapi.com/c854420/v854420774/9732f/7JeFZnZa-Xo.jpg" width=640 height=450>

## Project scheme
    .
    ├── model
    │   ├── unet.py             # defines U-Net class
    │   └── utils.py            # layers for U-Net class
    ├── tools                     
    │   └── data.py             # generates data 
    ├── main.py                 # main script to run
    └── ...

## Requirements

* Python 3.7
* Anaconda 3

## Installation

After installing the Python 3.7 version of [Anaconda](https://www.anaconda.com/distribution/), clone the project
```sh
git clone https://github.com/sevakon/unet-keras.git
```
Create and activate a tensorflow virtual environment (no GPU support):
```sh
conda create -n tensorflow_env tensorflow
conda avtivate tensorflow_env
```

Alternatively, create a tensorflow virtual environment with GPU support:
```sh
conda create --name tf_gpu tensorflow-gpu 
conda avtivate tf_gpu
```

Install other packages to the virtual environment:
```sh
conda install -c conda-forge keras   # installs Keras
```
```sh
pip install matplotlib               # installs matpoltlib
```
```sh
pip install scikit-image             # installs skimage
```

## Instructions
All the data preparation happens is data.py file. 
Data augmentation can be added to ImageDataGenerator parameters

```py
    image_datagen = ImageDataGenerator(rescale=1. / 255)      # add data augmentation here
    mask_datagen = ImageDataGenerator(rescale=1. / 255)       # as function parameter
```

Note that your data should be structures like this:

    .
    ├── train_path
    │   ├── img                 # training images
    │   │   ├── 1.jpg                       
    │   │   ├── 2.jpg                 
    │   │   └── ...    
    │   └── mask                # training masks
    │       ├── 1.png    
    │       ├── 2.png     
    │       └── ...    
    ├── test_path               # folder with images fot test
    │       ├── 1.jpg      
    │       ├── 2.jpg
    │       └── ...    
    ├── save_path               # where results of tests will be saved
    └── ...

  
Define your config in main.py file
```py
img_height =          # set your image resolution (512x512 or 256X256 recommended)
img_width =           # every image from your dataset will be resacled
img_size = (img_height, img_width)
train_path = ''       # full path to the folder with training data
test_path = ''        # full path to the folder with testing data
save_path = ''        # full path to the folder to save results on testing data
model_name = 'unet_model.hdf5'     # how to name your model save
model_weights_name = 'unet_weight_model.hdf5'
```

Run main.py script to generate data, fit model and predict results

## Results
After 10 epochs, accuracy of 98% was reached on a training set of 528 bone images. 
Testing results were satisfactory

Sample | Prediction
:-------------:|:-------------:
<img src="https://i.ibb.co/PYWQ5my/28.jpg">|<img src="https://i.ibb.co/5MWMRXH/28-predict.png">
<img src="https://i.ibb.co/8n3RZCm/6.jpg">|<img src="https://i.ibb.co/r5WWjJ9/6-predict.png">
<img src="https://i.ibb.co/hfrLpVY/4.jpg">|<img src="https://i.ibb.co/GPtgBpq/4-predict.png">

## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
