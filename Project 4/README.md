# Dog Breed Classifier
`The capstone project for Udacity’s Data Scientist Nanodegree Program`

### Table of Contents
1. [Project Overview](#overview)
2. [Problem Statement](#statement)
3. [File Descriptions](#desc)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Overview<a name="overview"></a>
In this project, I have developed an `end-to-end deep learning pipeline` that can analyze real-world, user-supplied photos **within a web or mobile app**. The pipeline will accept any user-supplied image as input and will predict whether a dog or human is present in the image. If a dog is detected in the image, it will provide an estimate of the dog’s breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 


## Problem Statement<a name="statement"></a>
In this project, I'm given RGB photographs of either humans and dogs, and I'm required to design and construct an algorithm that can identify either people or dogs in the images. If a dog or human is recognized, the algorithm must then determine its breed (if a dog is detected) or the breed that looks most like it (if a human is detected). The algorithm should prompt the user to provide a different image including either a dog or a human if neither is found in the image.


## File Descriptions<a name="desc"></a>
bottleneck_features

    | - DogResnet50Data.npz # ResNet-50 bottleneck features

haarcascades

    | - haarcascade_frontalface_alt.xml # pre-train weights for face detector

images # images for testing the final model

requirements # list all the libraries/dependencies required to run this project

saved_models # best weights of each CNN

    | - weights.best.from_scratch.hdf5
    
    | - weights.best.Resnet50_model.hdf5
    
    | - weights.best.VGG16.hdf5

dog_app.ipynb # main notebook of the project

extract_bottleneck_features.py # contains the code to use pre-trained imagenet models as **feature extractors** for transfer learning

report.html # the html file export from the notebook dog_app.ipynb

README.md


## Instructions<a name="instructions"></a>
1. Clone this github repository.
`git clone https://github.com/hoangth596/DS_Proj/tree/main/Project%204`

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and prepare image label pairs for training the model.

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and prepare images for the face detector model.

4. Open the notebook dog_app.ipynb to run the prediction model.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The credit is all go to Udacity for the project, the data and the notebook.