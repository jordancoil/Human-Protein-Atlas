import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import RepeatedKFold

import numpy as np
import pandas as pd

# Starting with an mnist implementation
# Later to repurpose for Human-Protein-Atlas

# List of things to implement
# 1. Training Labels array, filled - DONE
# 2. kfold class validation data splitter
# 3. Model Training Params class - DONE
# 4. Image Preprocessor - DONE
# 5. Image Batch Loader - DONE

from helpers.ModelParameters import ModelParameters
from helpers.TrainingDataLabelPreprocessor import TrainingDataLabelPreprocessor
from helpers.ImageHelper import ImageBatchGenerator, ImageLoader, ImagePreprocessor

# 6. F1 score
# 7. Prediction Generator Class
# 8. Model Class
# 9. Loss callback history tracking class
# 10. Train the model - save the model
# 11. Plot results (train loss, val loss, loss evolution per batch / per epoch)

# extras
# 1. Image Batch Loader - Implement target "wishlist" array


# Training params
training_data_path = 'train/train_images/'
modelParameters = ModelParameters(training_data_path)

# Preprocessing training data labels
training_data = pd.read_csv('train/train.csv')
trainingLabels = TrainingDataLabelPreprocessor(training_data)
trainingLabels.preprocess_data()
labeled_data = trainingLabels.training_data

imagePreprocessor = ImagePreprocessor(modelParameters)
imageLoader = ImageLoader(modelParameters)


# k-fold cross-validation
splitter = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)

partitions = []

for train_idx, test_idx in splitter.split(labeled_data.index.values):
    partition = {}
    partition["train"] = labeled_data.Id.values[train_idx]
    partition["validation"] = labeled_data.Id.values[test_idx]
    partitions.append(partition)

# Use the first partition to train the model
partition = partitions[0]


training_generator = ImageBatchGenerator(
    partition['train'],
    labeled_data,
    modelParameters,
    imagePreprocessor)
validation_generator = ImageBatchGenerator(
    partition['validation'],
    labeled_data,
    modelParameters,
    imagePreprocessor)
# predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, split between train and test sets
# We won't be able to load all our images into memory like the mnist set
# So we will replace this with an image batch processor
(x_train, y_train), (x_test, y_test) = mnist.load_data()

'''
The shape is (num_of_images_in_dataset, img_rows, img_cols, channels)
channels = 1 for greyscale
channels = 3 for RGB
For the Human-Protein-Atlas competition we will need channels = 4 because it uses RGBY
'''
x_train = x_train.reshape(
    60000,
    modelParameters.row_dimension,
    modelParameters.col_dimension, 1)
x_test = x_test.reshape(
    10000,
    modelParameters.row_dimension,
    modelParameters.col_dimension, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

'''
Convert class vectors to binary class matrices

eg. Here, the class value is converted into a binary class matrix
Y = 2 # the value 2 represents that the image has digit 2
Y = [0,0,1,0,0,0,0,0,0,0] # The 2nd position in the vector is made 1
'''
y_train = keras.utils.to_categorical(y_train, modelParameters.num_classes)
y_test = keras.utils.to_categorical(y_test, modelParameters.num_classes)

########################
# Start Building Model #
########################

# What is Sequential?
model = Sequential()

# what do "32" and "64" represent
# what is 'relu'
# what is kernel_size
# is (3, 3) in the second Conv2D also kernel_size?
# why does the second Conv2D have no input_shape
model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(
        modelParameters.row_dimension,
        modelParameters.col_dimension,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

# What does MaxPooling2D do?
# What does pool_size mean?
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout randomly switches off some neurons in the network
# This forces the network to find new paths and reduces overfitting
# This Dropout layer randomly switches off 25% of the neurons
model.add(Dropout(0.25))

# What does Flatten do?
model.add(Flatten())

# What does this Dense layer do?
# If the next dense layer predicts on the num_classes,
# does this Dense layer predict on 128 learned classes?
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# We can't use softmax because of imbalance
# we also cant use sigmoid (because reasons, see notes)
# what other options do we have? (answer will lie in other kernels or docs)
# Dense layers are for class prediction.
model.add(Dense(modelParameters.num_classes, activation='softmax'))

######################
# End Building Model #
######################


#######################
# Start Compile Model #
#######################

# We will be implementing the Focal loss function here
# What optimizer should we be using?
# I beleive our metric in this case is f1, not accuracyself.
# So how do we implement f1 score?
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#####################
# End Compile Model #
#####################


########################
# Start Training Model #
########################

# We should probably plot a graph of our metric over time
# to see how many epochs we should use.
# Our validation data should probably be a random k_fold validation set.
model.fit(x_train, y_train,
          batch_size=modelParameters.batch_size,
          epochs=modelParameters.num_epochs,
          verbose=1,
          validation_data=(x_test, y_test))

######################
# End Training Model #
######################

score = model.evaluate(x_test, y_test, verbose=0)
# What exactly is loss again?
print('Test loss:', score[0])
print('Test accuracy:', score[1])
