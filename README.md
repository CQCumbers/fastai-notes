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
3. Pre-trained model is VGG, which was created to classify ImageNet images into each of the thousands of categories.
  - Can run vgg on input by getting batches of training and validation sets, then using vgg fit() and finetune() to make the vgg model classify cats vs dogs. After doing this one can `save_weights()` to skip this step in future.
  - To submit to Kaggle, use Keras `predict_generator` to get all predictions and `batch.filenames` to get filenames. Use numpy savetxt() to save array of filenames and labels to csv (put this into convenience function). IPython FileLink allows one to easily download generated csv to local machine.

## Lesson 2
1. What is a neural network?
  1. Start with an input array (e.g. pixel array) and a target output array (e.g. 1-hot encoded category array)
    - 1-hot is where 1 = in category and 0 = not in category. Only one category has a 1 and everything else is 0
  2. Multiply input array by an array of random weights. Number of columns of weight array is up to programmer's discretion and part of designing the model architecture - more columns = more complex network.
    - Random weights often start with avg. 0 and variance 1/(n1 + n2), where n1 is length of input array and n2 is length of output array for this layer (this is Xavier initialization). Good initialization should have random weights creating output within an order of magnitude of target - this helps model find actual weights much faster.
  3. Repeat several times (several layers), though final output array must have same length as desired output array.     - Intermediate outputs are called activation layers. Nonlinear functions like relu (which is max(0,x)) are applied to intermediate outputs, and the nonlinear function output is fed to next layer (next matrix multiplication). With this one can approximate any function.
  4. Loss function is function that is higher the farther the function is from the target function (one that maps the input to the desired output). One loss function is avg of square errors.
  5. Optimize loss function by finding its derivative with respect to each of the weights. The sign of the derivative tells you whether to increase or decrease the weight to decrease the loss function. How much to increase or decrease the weight is the learning rate. `new_weight = sign( derivative of loss with respect to weight ) * learning rate`. One update of all weights = one iteration.
    - You can find the derivative for weights in earlier layers with chain rule. Say the loss function is `l(x, y)`, the neural network's first layer is `f(x, a)`, its second layer if `g(f, b)`, and its third layer is `h(g, c)`. The complete neural net is then `h(g(f(x, a), b), c)` where `a`, `b`, and `c` are weights. The derivative of the loss with respect to weight a is then `dl/da = dl/dh * dh/dg * dg/df * df/da`.
  6. Repeat many times, and you can find the minimum of the loss function. This optimization method is gradient descent.
    - For neural networks, there are so many parameters that best minimum is almost never found, so one optimizes until satisfied instead.
    - Stochastic gradient descent (SGD) is where loss function is evaluated using random subset (mini-batch) of data rather than entire set, and optimized to minimize that (technically stochastic is mini-batch of size 1, but general usage is stochastic = mini-batch). This is then repeated for all mini-batches in training data, for one epoch (# of mini-batches x # of epochs = # of times optimization is run = iterations). This makes it computationally feasible for neural networks to work and turns out not to matter vs regular gradient descent.
2. In Keras, you can create a linear model with a few lines of code.
  1. Create a linear model (which models the function `ax + b`) with `Dense()`
    - `activation='relu'` means after `ax + b` is done `max(0, x)` is applied to output.
  2. A one layer network (function is applied once to input) is `Sequential()`, with a single `Dense()` argument
  3. Compile with loss= loss function, like mean square error and optimizer = optimization method, like stochastic gradient descent with learning rate 0.01.
3. Common approach is to use a pretrained model's outputs as inputs for a linear model.
  - Pretrained model has already learned many useful low-level filters like circles, lines, curves, from a large data set. These filters will be useful for many tasks, especially those without as much data, so we reuse lower layers and finetune higher layers according to specific task.
  - To finetune, instead of using last layer output as input to linear model we can use second to last layer, so neural network uses learned features to calculate cats vs dogs rather than learn cats vs dogs from imagenet categories (which limits information).

## Lesson 3
1. Use [Deep Visualization Toolbox](https://github.com/yosinski/deep-visualization-toolbox) to see live updates of the convolutional filters developed by a CNN
    - Higher layers detect more complex features, cover a larger area
    - One does not need to determine what filters to develop manually, the CNN starts with random filters and finds the optimum set of filters to differentiate between the categories using the provided data.
2. Lesson 0 video and notebook cover convolutions
    - A convolutional filter (convolution) is a matrix that when multiplied by a subsection of an image matrix generates higher values when matching an interesting feature.
    - A convolution is a subset of a linear layer (all convolutions are linear layers), just without seperate parameters for each point (parameters are reused as filter slides across input).
    - `correlate() = convolve()` with 90 degree rotation (swap row-col)
3. Use `model.summary()` to view all layers of any keras model
    1. Max pooling simplifies images by replacing each n x n block in an image with the maximum pixel value in that block.
        - Enables filters built on filters - like top left and right of face filter match filter for eyes.
    2. Dealing with large images is as yet unsolved - 3 x 3 filters with more layers are considered optimal for detecting features in general.
        - Attentional models help, something that simulates foveation (fast eye movements) in an architecture might be long term solution
        - When not detecting something as similar to training data as cats v. dogs (like distracted drivers), remove more layers to get simpler and simpler filters that are more and more generally useful, then train your own specific layers from that point
            - Retraining convolution layers is not usually necessary for photo classification, as all useful spatial patterns are probably recognized by some set of Imagenet filters
        - Hard to finetune Imagenet models for cartoon images because Imagenet filters find photographic features with much higher frequency detail and texture, even in early filters.
    3. Another useful activation function (besides relu) is softmax
        - Used for last layer in network, matches well to 1-hot encoded output - squashes numbers to between 0 and 1, and ensure all numbers add up to 1, so output is like a probability distribution. This makes network faster to train.
    4. `None` in Ouput Shape means dependent on mini-batch size
4. What is overfitting and underfitting?
    - Underfitting: model has too few parameters/too little training data/is too simple to model function
        - Training set accuracy is lower than validation set accuracy
    - Overfitting: Model has fit the specific training set too well, rather than the general pattern
        - Training set accuracy is much higher than validation set accuracy
5. How to Reduce Overfitting:
    1. Add more data
        - Usually not an option with kaggle, try first when creating own data set
    2. Use data augmentation
        - Turns one data point into many data points through random transformations
        - Only augment and shuffle training set, validation set should not be modified
        - Augmented images should still look like a reasonable photo (don't stretch a photo of a cat so far that it doesn't look like a cat to a normal person)
        - Always augment data, question is just what kind of augmentation and how much
            - Keras has built in functions for randomly rotating, flipping, stretching, zooming, changing white balance, etc. of training images
    3. Use more generalizable architectures
        - Like batch normalization and convolutions.
    4. Regularization
        - Usually means using dropout
            - Dropout sets half of activations to 0 at random (throws away a certain percentage of that layer) so overfitting is harder
            - Reduce dropout to fix underfitting, increase to avoid overfitting to training data
            - Common approach is to set low dropouts (maybe 0.1) in early layers and higher (~0.5) dropouts in higher layers, with gradual increase in between. Dropout in earlier layers affects later layers.
            - Dropout is like random forest but for neural nets rather than decision trees - effectively constructions an ensemble of smaller neural nets at random
    5. Reduce architecture complexity (remove filters)
        - Last resort, try other options first and they will usually fix problem
6. Specific Techniques for Improving Accuracy:
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

## Lesson 4
1. What are convolutions?
    - Small (3x3 for images, usually) matrix, multiplied elementwise with every possible equivalently sized group in the input and the resulting matrix summed.
    - Multiple convolutions can be used in each layer, means multiplying input for that layer by each convolution's matrix to generate multiple matricies as the output. If n convolutions in a layer with p x p image as input, output is p x p x n.
    - Higher dimensional convolutions are used for later layers. 3 x 3 x n convolution multiplies each 3 x 3 region in the first input matrix with the first matrix of the convolution and adds that sum to the sum of the product of the second 3 x 3 matrix of the convolution and the corresponding 3 x 3 region of the second input.
    - Max pooling reduces the resolution by only allowing the maximum value in a certain (say 2 x 2) region to pass through. Applying this between convolutions may allow for larger scale features to be detected.
        - Some people prefer dilated convolutions (multiply convolution by every other pixel)
        - Max pooling helps with translation invariance, since features within 4 pixels are effectively the same
        - Reducing the resolution helps the convolutions cover larger area afterwards as well.
    - Dense layers equivalent to convolution with same dimensions as input (single window that doesn't slide)
    - See fast.ai excel spreadsheet for better visual of how this works
2. Comparing optimization algorithms
    1. SGD starts with a guess using an initialization procedure like Xavier
        - Finds derivative of loss function (like RMSE) for the mini-batch (loss function says how far the predicted points are from the actual points, and derivative says what direction to move parameters to decrease that)
            - relu doesn't technically have derivative at 0, but we can assume it to be 1 or 0 
        - Changes parameters in opposite of derivative direction by an amount proportional to the learning rate. 
            - If learning rate is too high, may constantly overshoot minimum and error will constantly increase.
        - Repeats an interation with a different mini-batch - once all possible mini-batches are evaluated an epoch is over. 
    2. Saddle points (think of descending river valley) are very common in practice but waste time with SGD because it swings from side to side across a valley, rather than going down the middle of the valley.
        - Using momentum means finding (exponentially weighted) average of past derivatives (average gradient) and using that to determine direction to move, rather than only using greatest gradient at that point. Solves saddle point problem.
            - Momentum hyperparameter (beta) says how much to weight previous step's direction versus current step's gradient. This is recursive (e.g. previous step's direction is 0.9 of this step's, but previous step's depended 0.9 on one before that, etc.)
    3. Different parameters converge at different rates (e.g. slope is found much faster than intercept) so dynamic learning rates can learn how fast the parameter converges and take different sizes steps for different parameters. 
        1. Adagrad takes average RMS of gradients (meaning derivative of loss) with respect to each parameter - higher if there are spikes in magnitude of gradients.  
            - More variation in gradients generates lower learning rates with Adagrad (as learning rate is divided by RMS). So learning rate parameter is usually set quite a bit higher with dynamic learning rates.
        2. RMSProp does exactly the same thing as momentum, but keeps track of square of last few gradients (basically magnitude) rather than gradient directions. Because of this the learning rate is based on exponentially weighted moving average rather than overall average as in adagrad, so it bounces around optimums rather than exploding, and can both increase and decrease the learning rate (adagrad only decreases)
        3. Adam exponentially weights previous momentums as well as previous directions (combining RMSProp and Momentum).
        4. Eve is an extension to Adam that adjusts the learning rate based on the exponentially weighted moving average of the rate of change of the loss function (with respect to the number of iterations rather than with respect to a parameter). If loss jumps around, learning rate decreases, if loss function stays constant, increases learning rate.
        5. Jeremy came up with something similar to Eve that adjusts the learning rate based on rate of change of the sum of squared errors of all parameters, for automatic learning rate decreases (never increases lr)
3. Tips from Jeremy's solutions to the State Farm competition
    1. Make sure to use sample set to decrease training times when experimenting
        - To see if sample size is too small, run validation on range of sample sizes and see the smallest one where the accuracy is reasonably similar to the accuracy over the entire validation set
    2. Batch normalize on first layer to avoid doing normalization as preprocessing.
    3. If loss doesn't decrease, perhaps it is jumping too far.
        - Common at the very start of training due to easy answers that work okay (like guessing same category and getting high accuracy for correctly saying its not in 9/10 classes)
        - Decrease the learning rate at the very start for 1-2 epochs, then increase it again
    4. Try very simple networks (single hidden dense layer) at first, to get baseline for whether a more complicated approach is working or not.
    5. Be careful not to jump to conclusions too quickly - make sure to run the model for enough epochs if the training loss is still decreasing.
    6. To find best data augmentation, try one at a time at 4 different levels, find the best one for each, then combine them together
    7. Regularization is strongly correlated with data set size (more data = less regularization) so experimentation with regularization cannot be done on sample.
    8. For kaggle remember to use clipping for best cross-entropy (prevent overconfident 1.0 and 0.0 predictions, clip to 0.98 and 0.02 for example)
    9. If you don't need to train the convolutional weights and you want to experiment only on the later dense layers, you can replace the original image data set with a data set consisting of the convolutional layers' predictions for that data. This way you can avoid retraining or even re-evaluating the filters and only retrain the dense layers (saves time with large image data sets)
        - Problem with precomputing some layers is that dynamic data augmentation does not work. Augmenting data set when calculating precomputed layers and predicting 5 times is an okay compromise.
4. Pseudo Labeling uses unlabeled data (test set) to help see structure of data (which ones are similar, etc.)
    - Forms of semi-supervised learning, where only some training data is labelled
    1. Use neural net to predict on test set, pseudo-labelling it.
    2. Use those pseudo-labels as if they were correct for training - can increase accuracy as neural network uses them for seeing structure.
        - Note that though validation set labels cannot be used for anything but validation, its inputs (images) and their generated pseudo-labels can be used for training. 
    3. Make sure only 1/4 - 1/3 of batches are pseudo-labelled data?
    4. You can keep using models trained on pseudo-labelled data to pseudo-label more data and so on.
5. Collaborative Filtering leads to Natural Language Processing
    - Used for recommender systems (Netflix, Amazon) and predicts a user's rating of a new item based on similar users' ratings for similar items.
    1. Think of categorizing a movie based on score in some categories (e.g. how sci-fi it is, how new it is, etc.) and categorizing users based on scores in same categories.
    2. Start with 5 random numbers for each movie and 5 random numbers for each user, and optimize for the dot products of these numbers for each combination of movie and user to be as close to the user's rating for that movie as possible.
        - These random numbers are latent factors, and by looking at the items (movies) with the highest value for a latent factor after training you can guess what that latent factor is detecting - sci-fi-ness, or Bruce-Willis-ness, etc. Check out netflix prize visualizations
    3. Add an additional bias number for each movie and user (accounts for a movie being more popular with all users or a user rating all movies more highly) which is added to the dot product.
    4. Embeddings map an integer to a column of a matrix. We can use them to match user id numbers to their corresponding latent factors. All of a user's latent factors together form a user vector.
    5. More powerful if instead of a dot product + bias function to get predicted ratings from latent factors, we used a neural network function. Same technique of optimizing latent factors as parameters, but with a more powerful function. Beats state of the art considerably on MovieLens.

## Lesson 5
1. Improving on Cats vs Dogs
    - Adding batchnorm to VGG improved results significantly - use VGG16BN from now on.
    - Batchnorm adds two more trainable parameters - a multiply and a bias - that can be learned, so if the model does need to make parameter much higher/lower it can adjust the standard deviation and mean of all parameters, avoiding large variances in scale between parameters and making the network more trainable.
2. Collaborative Filtering
    - Alternative regularization is l2
        - Adds sum of squares of weights to loss function, so function tries to minimize weights where it can. Prevents model from changing weights too much to avoid overfitting to training data.
    - Can just look at highest and lowest movie bias terms for more accurate view of best and worst movies
        - Removes noise due to reviewer favoritism of certain elements or consistently lower/higher scores from certain reviewers, as that is accounted for with latent factors and reviewer bias term, respectively.
    - Can use principal component analysis (PCA) to reduce number of latent factors, by combining ones that are most correlated (tend to move in the same direction). sklearn can do this for you - not that new.
        - Viewing movies with highest and lowest scores for certain latent factors (or combined latent factors) can tell you what that factor measures and its relative importance.
        - For movies, Jeremy finds "Classicness", "Big-Budgetness", and "Violence/Happiness" to be, in order, most important latent factors.
        - Important to visualize models to see what they are doing. Coefficients are not that useful, better to infer relations based on model results.
3. Keras Functional API
    1. Sequential models can only model linear flow of data through layers - cannot handle combining movie embeddings and user embeddings. For that we can use a model where each layer is a function and network is composition of functions. Functional API does this, and so can handle models with shared layers or multiple inputs (metadata) or outputs or directed acyclic graphs, etc.
    2. Each layer is a variable that is assigned the output of a function. The function is the layer type (e.g. `Dense()`) and it is immediately passed the variable containing the previous layer. `merge()` and other functions can then take multiple layers as input. First layer(s) should be of type `Input()`.
    3. Create model with `Model()`. When training the model we then need to pass an array if multiple inputs, and get out an array is multiple outputs.
4. Sentiment Analysis
    - imdb is a common dataset for this (built into keras), set of movie reviews labelled positive or negative
    1. Set goals and preprocess data
        - Academic accuracy is often found in 'Experiments' section of papers
        - Replace all rare words with same id to avoid having to learn lots of uncommon weights
        - Truncate and zero-pad reviews, as input data needs to all be same length. Keras has `pad_sequences()` to do this automatically.
        - Use sigmoid for last layer, to output 1/0 for binary classification (rather than softmax for 1-hot). Then use binary crossentropy for loss.
    2. Embedding is a lookup - equivalent to one-hot encoding and matrix multiply
        - Instead of embedding each movie/user, embed each word as combination of 32 "latent factors" (called a word vector)
        - Using dropout in the embedding `Embedding(dropout=0.2)` zeroes certain latent factors, while `Dropout(0.2)` afterwards zeroes certain words (entire rows)
            - Not much research on the exact effects of this, but using same dropout in and after embedding seems to work well.
        - Because words are often exactly the same (no differences in lighting or pose like in images), rather than pretrained networks for finetuning we can use pretrained word embeddings from large datasets. Basically never a good idea to start from random embeddings.
            - GLoVE and word2vec are common. GLoVE is trained on a variety of large data sets (as well as many versions), to get accurate global embeddings (all contexts). word2vec has good documenation on tensorflow.
            - Use cased and uncased versions depending on if capitalization is used. Tokenization (what is considered a word) is also important. Periods, "'s", can be included.
            - public word embeddings are usually obtained through unsupervised learning, where a fake task is developed whose labels are cheap to generate.
                - for word2vec, incoherent sentences are generated by replacing one word at random in a coherent sentence, and the network differentiates between coherent and incorrent sentences. Labels are already known because they are generated.
            - Relations between word embeddings can be very powerful (vector between 'man' and 'woman' embeddings equals vector between 'king' and 'queen' embeddings). This is possibly helped by how word2vec and GLoVE use simple linear models for training the embeddings.
            - You can plot 50-dimensional embeddings in 2D for visualization of groupings/distances with [tSNE](https://distill.pub/2016/misread-tsne/)
    3. In weight matrix, each 32-long row is a word vector and the 500 columns are every word in a review. Word order is learned from position along row.
    4. Use 1D convolutions (slides only along 1 dimension) to improve over single dense layer model.
        - Convolution matrices are really 2D, but have same row length as input, so it only slides along the columns
        - Just like how 2D convolution matrices on color images are really 3D, as they have 3 channels for color and don't slide in that direction.
        - You can use multiple convolution sizes (column lengths) in parallel with the functional API
            - Feed 3 sizes of convolution with the same input, then concatenate their output before passing it onto the next layer. You can then put the parallel layers within a sequential network.
5. Intuition for Recurrent Neural Networks (RNNs)
    - Used for tasks requiring memory, long-term dependencies, stateful representation, and variable sequence length
        - Used for visual attention to find house numbers, swiftkey word prediction, latex generation
    1. In sequential neural net output for one layer (intermediate output) is fed to input of next until we get final output.
    2. You can also merge intermediate output of parallel layers, for adding metadata information to the dense layers of a convnet or to use multiple size convolutions.
        - Merges are typically done by either summing or concatenating intermediate output matrices.
        - Predicting third word from two words can be done by applying layer operation on first word, then merging that intermediate output with second word, then applying another layer operation on the merged output.
        - You can keep merging intermediate outputs with another word more and more times to predict words based on more and more previous state.
    3. After generating new words from previous words, you can merge the generated words in as well. If you keep doing this as you generate more and more words, you have an RNN.
        - First, layer operation A is applied to input word 1 to get intermediate output X. Then, operation A is applied to input word 2 and operation B is applied to X, and X is updated to be the merged outputs of these operations. The process then repeats a certain number of times, before operation C is applied to the final X to generate the actual output. 
