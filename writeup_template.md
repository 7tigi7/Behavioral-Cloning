# **Behavioral Cloning** 
--
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Network Architecture

[Network Architecture](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)

CNN architecture.
The network architecture consists of 9 layers

First a 
Normalization layer
and 5
Convolutional layer
Convolutional layer
Convolutional layer
Convolutional layer
Convolutional layer
and finnaly 3
Fully connected layer
Fully connected layer
Fully connected layer

I trained the network weights to minimalize the error between the steering command output by the network.
The first layer of the network performs image normalization.
Normalization is a process that changes the range of pixel intensity values.
This important step is ensures that each input image has a similar data distribution.
Convolutional layers are performed on input data with the use of a filter or kernel to produce a feature map.
I use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, 
and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.
The five convolutional layers followed by with three fully connected layers, leading to a final output control value. 
The fully connected layers are designed to function as a controller for steering.
The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (model.py line 71). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 79,85). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 67-97). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 94).
[Adam Optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find a good solution to drive a car on the road in the simulator.
My first step was to use a convolution neural network model as described above. I thought this model might be appropriate because NVIDIA is also used it for selfdriving cars.


To combat the overfitting, I only changed that i set a dropout at (model.py) lines 79,85
Read more about dropout
[Dropout](https://tf-lenet.readthedocs.io/en/latest/tutorial/dropout_layer.html)

Then I compiled and finalized the model and saved it into the *model.h5* file.
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track so to improve the driving behavior in these cases, I made more data to the training and tried to drive the car in the center of the road while recording.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 67-97)
Here is a visualization of the architecture
[CNN architecture](cnn-architecture-624x890.png)

#### 3. Creation of the Training Set & Training Process

I recorded the vehicle recovering from the left side and right sides of the road back to center.
Then I repeated this process on track two in order to get more data points.
To augment the data sat, I also cropped images.
Keras provides the Cropping2D layer for image cropping within the model. This is relatively fast, because the model is parallelized on the GPU, so many images are cropped simultaneously.
By contrast, image cropping outside the model on the CPU is relatively slow. The Cropping2D layer might be useful for choosing an area of interest that excludes the sky and/or the hood of the car.

Here is an example of an input image and its cropped version after passing through a Cropping2D layer:

[Cropping](cropped.jpg)

Size of data is 8036.
These are images from left right and the center camera of the simulator and the angles to each one.
The log csv file
[Log file](drivinglog.jpg)

I add the pictures more than once to the training data set so the net can learn more.
I finally randomly shuffled the data set and trained the net on 72324 samples.
The samples size is very lagre and the epoch size is 30 so the calculation is take more time but the result is better.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here's a [link to my result video ](./video.mp4)

### Discussion

In this project there was a lot of fun and I learned about keras and i was attended to make a real working neural network with it.
Neural networks are sets of algorithms intended to recognize patterns and interpret data through clustering or labeling. 
In other words, neural networks are algorithms. A training algorithm is the method you use to execute the neural network's learning process.
Keras is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning models. 
It wraps the efficient numerical computation libraries Theano and TensorFlow and allows you to define and train neural network models in just a few lines of code.

Here are some links related to this Udacity Behavioral Cloning project
[Neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
[Machine Learning introduction](https://medium.com/cracking-the-data-science-interview/a-gentle-introduction-to-neural-networks-for-machine-learning-d5f3f8987786)
[Keras](https://keras.io/)
[NVIDIA deep learning](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)
