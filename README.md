# In-class Intel Image Classification using Convolutional NNs
Competition link: https://www.kaggle.com/t/5aee1022db0942818b00a31000c1f09a

# For MODEL RESULTS see the PDF File

# Overview of the assignement:
The objective of this assignment is to enhance your understanding of Convolutional Neural Networks (CNNs).
The assignment is to experiment with CNNs and write about your experience on model building and hyperparameter tuning and the analysis of results.
The CNN experimentation part is made into a Kaggle inClass competition (https://www.kaggle.com/t/5aee1022db0942818b00a31000c1f09a).
The competition is to classify the dataset called 'Intel Image Classification'.  
This Data contains a total of around 17k color images of size 150x150 pixels, distributed over 6 categories: {'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'}.  
In this assignment, you will use CNNs to classify the images. 
The dataset is already divided into training and test sets. 
You use the training set to train a CNN model (though you further split the training set to training and validation subsets in the code). 
After training the model, you generate predictions for the test set and submit the predictions to the competition.

# Model Development and Hyperparameter Tuning:

The starter code includes a simple CNN model (which is not included in repo). I was tasked with experimenting with models/hyperparameters to find an optimal configuration.
IMPORTANT: the assignment didn't allow to use a pre-trained network (such as VGG net or MobileNet) -- I had to train the model from scratch in this assignment or competition.  However Data Augmentation was allowed.

Parameters related to models which were tuned included:  

number of filters.
size of filters
number of Convolution layers.
number and size of Fully Connected layers.
use of Dropout layers and the percentages.
use of regularization.
use of BatchNormalization layers. Look at the Keras Documentation to learn about it.

Hyperparameters related to compilation which were tuned include:

learning rate
regularization
drop-out percent
mini-batch size 

# Evaluation
The evaluation metric for this competition is multi-class logarithmic loss.
where N is the number of images in the test set, M is the number of image class labels and ln is the natural logarithm. y_ij is 1 if observation i belongs to class j and 0 otherwise, and p_ij is the predicted probability that observation i belongs to class j. Since each image is labeled with exactly one true class, in this dataset, the metric is equivalent to LogLikelihood.
The LogLoss values are to be minimized. Smaller values, towards 0, indicate better performance (because LogLoss is always non-negative).
Note that the code uses categorical_crossentropy as the loss function, while Kaggle evaluation uses log_loss (which is the same as 'log likelihood').  So the performance results on Kaggle may differ from what you get during training.
I measured the performance also by other aspects besides the competition evaluation metric, such as model complexity, computational speed and learning stability. Consider the trade-off between those aspects vs. performance.

# Submission Format
For each image in the test_pred set, you predict the probability of it belonging to each class. Your submission file should be in .csv format: comma-separated values with no extra spaces. The file should contain a header row, that is, a line with column labels, and have exactly the format shown below. Your numbers, except for the first column, should be different from these.
