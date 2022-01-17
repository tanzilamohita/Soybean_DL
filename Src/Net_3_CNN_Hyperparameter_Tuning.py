# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 12/26/2021
# ===============================
import tensorflow
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
import os

# import glob
# import matplotlib.pyplot as plt
import pandas as pd
# from natsort import natsorted
# from keras.utils.vis_utils import plot_model
# from sklearn.metrics import mean_squared_error, make_scorer
import time


# keras items
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D, LSTM #CNNs
from tensorflow.python.keras.layers import LeakyReLU

# using GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# model will be trained on GPU 0,
# if it is -1, GPU will not use for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load Processed Data
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

start_time = time.time()
Y = pd.read_csv('../Data_Prediction/Soybean_Y.csv', index_col=0)
missing_values_Y = Y.isnull().sum().sum()
print("Total Missing values in Y:", missing_values_Y)
Y = Y.dropna()
Y.columns = np.arange(0, len(Y.columns))
print("Y shape", Y.shape)

X = pd.read_csv('../Data_Prediction/Compress_3_Chunk/Soybean_X_C3_0.csv', index_col=0,
                low_memory=False)
X = X.T
X.columns = np.arange(0, len(X.columns))
X = X.drop(X.index.difference(Y.index))
Y = Y.drop(Y.index.difference(X.index))
missing_values_X = X.isnull().sum().sum()
print("Total Missing values in X:", missing_values_X)
print("X shape", X.shape)
print("Y shape", Y.shape)


# data partitioning into train and validation
itrait = 0  # first trait analyzed
# print(Y[itrait])
X_train, X_valid, y_train, y_valid = train_test_split(X, Y[itrait], test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

X2_train = np.expand_dims(X_train, axis=2)
X2_valid = np.expand_dims(X_valid, axis=2)


def create_model(loss, learning_rate):
    nSNP = X_train.shape[1]
    nStride = 2  # stride between convolutions

    # Instantiate
    model_cnn = Sequential()
    # add convolutional layer
    model_cnn.add(Conv1D(32, nStride, activation="relu", input_shape=(nSNP, 1)))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(Dropout(0.5))
    # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))
    # add convolutional layer
    model_cnn.add(Conv1D(64, nStride, activation="relu"))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(Dropout(0.5))
    # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))
    # add convolutional layer
    model_cnn.add(Conv1D(128, nStride, activation="relu"))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(Dropout(0.5))
    # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))

    # Solutions above are linearized to accommodate a standard layer
    model_cnn.add(Flatten())
    model_cnn.add(Dense(64, activation='linear'))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(Dense(1))

    adm = keras.optimizers.Adam(learning_rate=learning_rate)
    # Model Compiling (https://keras.io/models/sequential/)
    model_cnn.compile(loss=loss, optimizer=adm)
    return model_cnn


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
param_grid = dict(batch_size=batch_size, epochs=epochs, loss=loss, learning_rate=learning_rate)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
# Fit grid search
grid_result = grid.fit(X2_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f with: %r" % (mean, param))


print('Total Time: ', time.time() - start_time)