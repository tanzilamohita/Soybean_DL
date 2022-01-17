# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/3/2022
# ===============================
# main modules needed
from numpy.random import seed
# seed(1)
import tensorflow
# tensorflow.random.set_seed(2)

import pandas as pd
import numpy as np

import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from tensorflow import keras
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, logcosh
from tensorflow.keras.utils import plot_model
import os
import time
import pathlib


# keras items
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D, LSTM #CNNs
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LeakyReLU


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

start_time = time.time()
Y = pd.read_csv('../Data_Prediction/Soybean_Y.csv', index_col=0)
missing_values_Y = Y.isnull().sum().sum()
print("Total Missing values in Y:", missing_values_Y)
Y = Y.dropna()
Y.columns = np.arange(0, len(Y.columns))

def CNN(data):
    # Instantiate
    model_cnn = Sequential()
    # add convolutional layer
    model_cnn.add(Conv1D(32, 2, activation="relu", input_shape=(data, 1)))
    # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))

    # # add convolutional layer
    model_cnn.add(Conv1D(64, 2, activation='relu'))
    # # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))
    # # add convolutional layer
    model_cnn.add(Conv1D(128, 2, activation='relu'))
    # # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))

    # Solutions above are linearized to accommodate a standard layer
    model_cnn.add(Flatten())
    model_cnn.add(Dense(64, activation='linear'))
    # model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(Dense(1))

    adm = keras.optimizers.Adam(learning_rate=0.0001)
    # Model Compiling (https://keras.io/models/sequential/)
    model_cnn.compile(loss='mse', optimizer=adm)
    #
    # # list some properties
    model_cnn.summary()
    return model_cnn

trait_loss = []
trait_corr = []
itrait = 0
for itrait in range(0, 3):
    chunk_num = 0
    chunk_corr = []
    chunk_loss = []
    chunk = pd.read_csv('../Data_Prediction/Soybean_X_C3.csv', index_col=0,
                        low_memory=False, chunksize=5000)
    for X in chunk:
        X = X.T
        X.columns = np.arange(0, len(X.columns))
        X = X.drop(X.index.difference(Y.index))
        Y = Y.drop(Y.index.difference(X.index))
        missing_values_X = X.isnull().sum().sum()
        print("Total Missing values in X:", missing_values_X)
        print("X shape", chunk_num, itrait, X.shape)
        print("Y shape", chunk_num, itrait, Y.shape)

        X_train, X_valid, y_train, y_valid = train_test_split(X, Y[itrait], test_size=0.2)
        print("X_train shape and y_train shape", X_train.shape, y_train.shape)
        print("X_valid shape, y_valid shape", X_valid.shape, y_valid.shape)

        batch_size = 128
        epochs = 50
        nSNP = X_train.shape[1]
        print("SNP", nSNP)
        nStride = 2  # stride between convolutions

        X2_train = np.expand_dims(X_train, axis=2)
        X2_valid = np.expand_dims(X_valid, axis=2)

        print("X2_train shape", X2_train.shape)
        print("X2_valid shape", X2_valid.shape)

        model_cnn = CNN(nSNP)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        # # training
        model_cnn_train = model_cnn.fit(X2_train, y_train, epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(X2_valid, y_valid),
                                        shuffle=True, callbacks=[es])

        # print(model_cnn_train.history)
        loss = model_cnn_train.history['loss']
        val_loss = model_cnn_train.history['val_loss']

        # cross-validation
        mse_prediction = model_cnn.evaluate(X2_valid, y_valid, batch_size=batch_size)
        print('\nMSE in prediction =', chunk_num, itrait, mse_prediction)
        chunk_loss.append(mse_prediction)

        # get predicted target values
        y_hat = model_cnn.predict(X2_valid, batch_size=batch_size)
        np.seterr(divide='ignore', invalid='ignore')

        # correlation btw predicted and observed
        corr = np.corrcoef(y_valid, y_hat[:, 0])[0, 1]
        print('\nCorr obs vs pred =', chunk_num, itrait, corr)
        chunk_corr.append(corr)
        chunk_num += 1

    print("Chunk Loss: ", chunk_loss)
    print("Chunk Corr: ", chunk_corr)
    trait_loss.append([itrait, chunk_loss])
    trait_corr.append([itrait, chunk_corr])
    itrait += 1

print("Trait Loss: ", trait_loss)
print("Trait Corr: ", trait_corr)
print('Total Training Time: ', time.time() - start_time)




