# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 10/14/2021
# ===============================

import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras import Input, Model
from keras.layers import Dense
import numpy as np
import os
import keras
import glob
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error, make_scorer
import time
import pathlib
from numpy import mean
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
# using GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# model will be trained on GPU 0,
# if it is -1, GPU will not use for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load Processed Data
fileList = natsorted(glob.glob("../Data/Split_Input_Data/"+"*.csv"))
fileList = fileList[0:1]
print(fileList)

start_time = time.time()

index = 0
x_train = 0
x_test = 0
x_valid = 0

for filename in fileList:
    digital_data = pd.read_csv(filename, low_memory=False, header=None)
    digital_data = np.array(digital_data)
    # print(index, digital_data)

    # Creating a training set, test set and validation set
    x_train, x_mid = train_test_split(digital_data, test_size=0.4)
    x_test, x_valid = train_test_split(x_mid, test_size=0.5)
    print('Shape of train data: ', x_train.shape)
    print('Shape of test data: ', x_test.shape)
    print('Shape of validation data: ', x_valid.shape)
    index += 1


def create_model(loss, optimizer, activation, neuron1=1, neuron2=1, neuron3=1):
    # Taking the input data of dimension 28 and convert it to keras tensors.
    input = Input(shape=(digital_data.shape[1],))
    print('Shape of input layer data: ', input.shape)

    # For all the hidden layers for the encoder and decoder
    # we use relu activation function for non-linearity.
    encoded = Dense(neuron1, activation=activation)(input)
    encoded = Dense(neuron2, activation=activation)(encoded)
    encoded = Dense(neuron3, activation=activation)(encoded)

    decoded = Dense(neuron2, activation=activation)(encoded)
    decoded = Dense(neuron1, activation=activation)(decoded)
    # The output layer needs to predict the probability of an output
    # which needs to either 0 or 1 and hence we use sigmoid activation function.
    decoded = Dense(digital_data.shape[1], activation='sigmoid')(decoded)

    # this model maps an input to its reconstruction
    # creating the autoencoder with input data. Output will be the final decoder layer.
    autoencoder = Model(input, decoded)
    # extracting the encoder which takes input data and the output of encoder
    # encoded data of dimension 3
    encoder = Model(input, encoded)
    # the structure of the deep autoencoder model
    # autoencoder.summary()
    # the structure of the encoder
    # encoder.summary()
    #adm = tf.keras.optimizers.Adam(learning_rate=0.001)
    # We use MSE to calculate the loss of the model
    autoencoder.compile(optimizer=optimizer, loss=loss)
        # We finally train the autoencoder using the training data with 100 epochs and batch size of 52.
    return autoencoder


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# create model
model = KerasRegressor(build_fn=create_model, verbose=0)
# define the grid search parameters
optimizers = ['SGD', 'RMSProp', 'Adam']
batch_size = [4, 8, 16, 32, 52, 64, 128, 256, 450]
epochs = [50, 100, 150, 200, 250, 300, 350, 400]
learning_rate = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
activation = ['relu', 'sigmoid']
loss = ['mean_squared_error', 'binary_crossentropy', 'mean_absolute_error']
neuron1 = [14, 15, 16, 17, 18, 19, 20]
neuron2 = [7, 8, 9, 10]
neuron3 = [1, 2, 3, 4, 5]
param_grid = dict(batch_size=batch_size, epochs=epochs, loss=loss, optimizer=optimizers,
                  activation=activation, neuron1=neuron1, neuron2=neuron2, neuron3=neuron3)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
# Fit grid search
grid_result = grid.fit(x_train, x_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f with: %r" % (mean, param))



print('Total Time: ', time.time() - start_time)

