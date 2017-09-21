# Notes for couse.fast.ai MOOC (Part 1)

## AWS Guide
1. Use  `setup-p2.sh` from setup folder to create a new p2 instance. Beforehand, ensure AWS limit allows at least one p2 instance, following setup video. Follow setup video for AMI and needed software installations as well. Only do this step once.
2. Run `source aws-alias.sh`, then `aws-start` to start up p2 instance
3. Run `aws-get-p2` and `aws-ip` to store instance id and ip.
4. Connect to instance via ssh with `aws-ssh`, and start jupyter notebook with `jupyter notebook`
5. Point web browser to notebook ip with `aws-nb`. To return to local terminal session, use `C-a d` in remote tmux session and `exit` ssh.
6. Terminate spot session (to save money) with `aws ec2 terminate-instances --instance-ids $instance-id` (substitude $instance-id according to step 2)

## Lesson 1
1. Kaggle is machine learning competitions and data sets
  - Top 50%: okay-ish model, Top 20%: good model, Top 10%: expert for this kind of problem, Top 10: Best in the world
2. Data organization is very important.
  - Split into training set and test set - do not view test set until finished.
  - Split training set to training and validation sets.
  - Copy tiny (~100 file) sample of training and validation sets to sample training and validation set, in order to prototype and test script working quickly.
  - Split training and validation set into different directories for each category to classify - e.g. `dogs` and `cats`.
3. Pre-trained model is Vgg, which was created to classify ImageNet images into each of the thousands of categories.
  - Can run vgg on input by getting batches of training and validation sets, then using vgg fit() and finetune() to make the vgg model classify cats vs dogs. After doing this one can `save_weights()` to skip this step in future.
  - To submit to Kaggle, use Keras `predict_generator` to get all predictions and `batch.filenames` to get filenames. Use numpy savetxt() to save array of filenames and labels to csv (put this into convenience function). IPython FileLink allows one to easily download generated csv to local machine.

## Lesson 2
1. What is a convolutional neural network?
  1. Start with an input array (e.g. row of pixels) and a target output array (e.g. category array where 1 = in category 0 = not in category)
  2. Multiply input array by an array of random weights, usually with avg. 0 and variance 1/(n1 + n2), where n1 is length of input array and n2 is length of output array for this layer (this is Xavier initialization, others exist). Number of columns of weight array is up to programmer's discretion and part of designing the model architecture - more columns = more complex network.
  3. Repeat several times (several layers), though final output array must have same length as desired output array. Good initialization should have random weights creating output within an order of magnitude of target.
    - Intermdeiate outputs are called activation layers. Nonlinear functions like relu (which is max(0,x)) are applied to intermediate outputs, and the nonlinear function output is fed to next layer (next matrix multiplication). With this one can approximate any function.
  4. Loss function is function that is higher the farther the function is from the target function (one that maps the input to the desired output). For example, avg of square errors.
  5. Optimize loss function by finding its derivative with respect to each of the weights. The sign of the derivative tells you whether to increase or decrease the weight to decrease the loss function. How much to increase or decrease the weight is the learning rate.
    - You can find the derivative for composition of function (e.g. multiple layers) with chain rule - first find derivative of last function with respect to its inputs, then multiply it by derivative of second-to-last function with respect to its inputs, then so on until we get derivative of loss function with respect to original inputs (weights). This is backpropogation.
  6. Repeat many times, and you can find the minimum of the loss function. This optimization method is gradient descent.
    - For neural networks, there are so many parameters that best minimum is almost never found, so one optimizes until satisfied instead.
    - Stochastic gradient descent (SGD) is where loss function is evaluated using random subset (mini-batch) of data rather than entire set, and optimized to minimize that (technically stochastic is mini-batch of size 1, but general usage is stochastic = mini-batch). This is then repeated for all mini-batches in training data, for one epoch (# of mini-batches x # of epochs = # of times optimization is run = iterations). This makes it computationally feasible for neural networks and turns out not to matter vs regular gradient descent.
2. In Keras, you can create a linear model with fewer lines of code.
  1. Create a linear model (function is ax+b, where x is input array) with Dense()
    - activation= relu means after ax+b is done max(0,x) is applied to output.
  2. A one layer network (function is applied once to input) is Sequential(), with a single Dense() argument
  3. Compile with loss= loss function, like mean square error and optimizer = optimization method, like stochastic gradient descent with learning rate 0.01.
3. Common approach is to use a pretrained model's outputs as inputs for a linear model.
  - Pretrained model has already learned many useful low-level filters like circles, lines, curves, from a large data set. These filters will be useful for many tasks, especially those without as much data, so we reuse lower layers and finetune higher layers according to specific task.
  - To finetune, instead of using last layer output as input to linear model we can use second to last layer, so neural network uses learned features to calculate cats vs dogs rather than learn cats vs dogs from imagenet categories (which limits information).

## Lesson 3
1. Use [Deep Visualization Toolbox](https://github.com/yosinski/deep-visualization-toolbox) to see live updates of the convolutional filters developed by a CNN
    - Higher layers detect more narrowly defined but more complex things, cover a larger area
    - One does not need to determine what filters to develop manually, the CNN starts with random filters and gradually optimizes (using SGD) for the best set of filters to differentiate between the provided categories, with the provided data.
2. Lesson 0 video and notebook cover convolutions
    - Convolution is a filter, a matrix that when multiplied by a subsection of an image matrix, generates higher values when matching an interesting feature.
    - A convolution is a subset of a linear layer (that is all convolutions are linear layers) so every few layers in a neural network one can substitute a linear model for a convolution
3. Use model.summary() to view all layers of any keras model
    - Max polling simplifies images by replacing each nxn block in an image with the maximum pixel value in that block.
        - Enables filters built on filters - like top left and right of face filter match filter for eyes
    - Hard to fintune Imagenet models for cartoon images because Imagenet filters find photographic features with much higher frequency detail and texture, even in early filters.
    - correlate() = convolve() with 90 degree rotated filters (swap row-col)
4. Another useful activation function besides relu is softmax
![softmax formula](https://cloud.githubusercontent.com/assets/14886380/22743247/9eb7c856-ee54-11e6-98ca-a7e03120b1f8.png) 
    - Used for last layer in network, matches well to 1-hot encoded output - squashes numbers to between 0 and 1, and ensure all numbers add up to 1, so output is like a probability distribution
    - Any architecture large enough can do any task, but different architectures are faster at certain tasks.
        - Softmax is easy to convert to 1-hot and so faster to train
    - Dealing with large images is as yet unsolved - 3x3 filters with more and more layers are considered optimal for detecting features in general
        - Attentional models help, something that simulates foveation (fast eye movements) in an architecture might be long term solution
        - When not detecting something as similar to training data as cats v. dogs (like distracted drivers), remove more layers to get simpler and simpler filters that are more and more generally useful, then train your own specific layers from that point
            - Retraining convolution layers is not usually necessary for photo classification, as all useful spatial patterns are probably recognized by some set of Imagenet filters
5. Avoiding overfitting and underfitting
    - Underfitting: model has too few parameters/too little training data/is too simple to model function
        - Training set accuracy is lower than validation set accuracy
    - Overfitting: Model has fit the specific training set too well, rather than the general pattern
        - Training set accuracy is much higher than validation set accuracy
    - Several methods to deal with overfitting. In order:
        1. Add more data
            - Usually not an option with kaggle, try first when creating own data set
        2. Use data augmentation
            - Turns one data point into many data points through random transformations
                - Only augment and shuffle training set, validation set should not be modified
                - Augmented images should still look like a reasonable photo (don't stretch a photo of a cat so far that it doesn't look like a cat to a normal person)
                - Always augment data, question is just what kind of augmentation and how much
                    - Keras has built in functions for randomly rotating, flipping, stretching, zooming, changing white balance, etc. of training images
        3. Use more generalizable architectures
        4. Regularization
            - Usually means using dropout
                - Dropout sets half of activations to 0 at random (throws away a certain percentage of that layer) so overfitting is harder
                - Reduce dropout to fix underfitting, increase to avoid overfitting to training data
                - Common approach is to set low dropouts (maybe 0.1) in early layers and higher (~0.5) dropouts in higher layers, with gradual increase in between. Dropout in earlier layers affects later layers.
                - Dropout is like random forest but for neural nets rather than decision trees - effectively constructions an ensemble of smaller neural nets at random
        5. Reduce architecture complexity (remove filters)
            - Last resort, try other options first and they will usually fix problem
    - Batch normalization is used in almost all modern deep learning - almost always a good idea
        - Converts data into z-scores
        - Allows weights to affect outcome more evenly on different scale (order of magnitude) data
        - Keras model preprocess subtracts mean from data to get closer to normalized values
        - 10x faster than not using it and reduces overfitting without throwing away data
        - Allows SGD to change scale of all weights, rather than sometimes only changing magnitude of 1 and making model fragile
    - Ensembling
        - Usually a surefire way to improve model accuracy a bit, but time consuming
        - Train a set (maybe ~6) of the same or similar models, will have different errors due to starting from different intial random parameters
        - Take average of set predictions for each validation set image
