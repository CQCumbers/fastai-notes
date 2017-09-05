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

