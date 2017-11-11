#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./new-images/histogram.png "Visualization"
[image2]: ./new-images/grayscale.jpg "Grayscaling"
[image3]: ./new-images/random_noise.jpg "Random Noise"
[image4]: ./new-images/image0.jpg "Traffic Sign 1"
[image5]: ./new-images/image1.jpg "Traffic Sign 2"
[image6]: ./new-images/image2.jpg "Traffic Sign 3"
[image7]: ./new-images/image3.jpg "Traffic Sign 4"
[image8]: ./new-images/image4.jpg "Traffic Sign 5"
[image9]: ./new-images/image_input.png "image_input"
[image10]: ./new-images/visualize_cnn.png "visualize_cnn"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many traffic signs in each label in the training set

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it help to reduce the training time. I think that grayscale images could adapt to different illumination conditions

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it made more easy to train using a single learning rate. The dataset mean was reduced from 82 to -0.35. 

Here is an example of an original image and an augmented image:

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Dropout               | keep_prob=0.5 for training                    |
| Convolution 3x3 		| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Dropout               | keep_prob=0.5 for training                    |
| Convolution 3x3 		| 1x1 stride, valid padding, outputs 12x12x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64 				    |
| Dropout               | keep_prob=0.5 for training                    |
| Convolution 3x3 		| 1x1 stride, valid padding, outputs 4x4x128 	|
| RELU					|												|
| Dropout               | keep_prob=0.5 for training                    |
| Flatten				| flat 14x14x32 to 6272, flat 6x6x64 to 2304, flat 4x4x128 to 2048|
| Concat flat			| Concat 6272, 2304 and 2048 to 10624			|
| Fully connected 		| 10624 to 1024                             	|
| RELU                  |                                               |
| Dropout               | keep_prob=0.5 for training                    |
| Fully connected 		| 1024 to 1024                               	|
| RELU                  |                                               |
| Dropout               | keep_prob=0.5 for training                    |
| Fully connected 		| 1024 to 43                                	|
 
####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the 3-stage ConvNet architecture based on the paper of Sermanet. I chosed a learing rate of 1e-3, batch size of 128, 60 epoches, mu of 0, sigma of 0.1. I used adamoptimizer for optimization. The training parameters of 3-stage ConvNet architecture were about 220 times more than the LeNet-5 architecture. if I trained the model on my computer using CPU, it took about a few hours. So I trained the model on floydhub using GPU, the optimization process took only about 724 seconds.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.967
* test set accuracy of 0.972

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  The LeNet-5 architecture was tried. 
* What were some problems with the initial architecture?
  The accuracy of on test data set is much lower than the accuracy of validation data set and test data set. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  The LeNet-5 architecture model is over fitting. I added dropout to the model. I am not sure where should dropout be added, it did improve the accuracy of test data set. I added layers using 3-stage ConvNet architecture in the paper of Pierre Sermanet, I concat features from multiple layers. Based on the paper, the first stage extracts "local" motifs with more precise details, the second and third stage extracts "global" and invariant shapes and structures. 
 
* Which parameters were tuned? How were they adjusted and why?
  The epoches, batch_size, learning rate and keep_prob are tuned. By addind and subtracting the parameters, The test accuracy is used to identify which parameters are better.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  The images were pre-processed by grayscaling and normalization, which reduced the training time and made more easy to train using a single learning rate. Dropout was also needed to avoid over-fitting. I used 3-stage ConvNet architecture to extract features from multiple layers

If a well known architecture was chosen:
* What architecture was chosen?
  The 3-stage ConvNet architecture was chosen.
* Why did you believe it would be relevant to the traffic sign application?
  The LeNet could be successfully applied to MNIST data set to identify handwriting characters. The traffic signs images are same as handwriting characters with surrounding pixels.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  Both the validation and test accuracy is almost 1.0. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because the illumination was too dark to identify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop      		    | Stop  								    	| 
| Speed limit (50km/h)  | Speed limit (50km/h)							|
| Yield					| Yield											|
| Traffic signals	   	| Traffic signals					 			|
| Turn right ahead		| Turn right ahead      					    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. During training, the accuracy of testing data set is 0.939. Considering the fact that only 5 sample images used for test, the accuracy is reasonable.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| Stop   									| 
| 2.02027731e-24     				| Speed limit (30km/h)										|
| 1.44339692e-25 					| No entry											|
| 6.89954311e-27	      			| General caution					 				|
| 3.54251471e-27				    |Turn left ahead     							|


For the second image, the model is relatively sure that this is speed limit (50km/h) sign (probability of 9.99881864e-01), and the image does contain speed limit (50km/h) sign. The top five soft max probabilities were 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| Speed limit (50km/h)   				|
| 2.32053221e-08     				| Speed limit (80km/h)						|
| 8.69083294e-09					| Speed limit (30km/h)					|
| 6.39256825e-09	      			| Speed limit (60km/h)				 				|
| 6.47225275e-13				    |Speed limit (100km/h)  							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is an example of an original input image :

![alt text][image9]

Here is the output feature maps of the second convolutional layer :

![alt text][image10]




