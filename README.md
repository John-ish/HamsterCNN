# HamsterCNN
An Emotion Detection Convolutional Neural Network that displays images of a silly little guy based on the emotion.

This is my first ever attempt at making a CNN. This was a fun little project I worked on to understand the concept of CNNs 

The CNN uses the FER-2013 dataset from kaggle.

The dataset was downloaded and loaded through the file path in the script so make sure the path of the dataset and the hamster images point to the right file. 

Ensure to update the placeholders in the file path variables before running the script. 

The script loads in the CNN model from a json file using the model_from_json library from keras. 
The model from the json file is the one I've trained. 
To train your own model refer to the TrainModel.py file which creates a json file of the model and a file with the weights.
Simply update the file paths in the CNN-Practice.py file to use the model you've created.

I've used this approach as you can train multiple models and have different weights files and .json files of these models to reuse on the CNN-practice.py file. (Yes, I am aware this is probably not very efficient.) 
