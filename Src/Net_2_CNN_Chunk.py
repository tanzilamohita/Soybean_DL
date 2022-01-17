# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/17/2022
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
from natsort import natsorted
import os
import time
import glob
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
    model_cnn.add(Conv1D(96, 3, activation="relu", input_shape=(data, 1)))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(Dropout(0.5))
    # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))

    # # add convolutional layer
    model_cnn.add(Conv1D(192, 3, activation='relu'))
    # model_cnn.add(LeakyReLU(alpha=0.1))
    # model_cnn.add(Dropout(0.5))
    # # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))
    # # add convolutional layer
    model_cnn.add(Conv1D(32, 3, activation='relu'))
    # model_cnn.add(LeakyReLU(alpha=0.1))
    # model_cnn.add(Dropout(0.5))
    # # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))

    # # add convolutional layer
    model_cnn.add(Conv1D(32, 3, activation='relu'))
    # model_cnn.add(LeakyReLU(alpha=0.1))
    # model_cnn.add(Dropout(0.5))
    # # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))

    # Solutions above are linearized to accommodate a standard layer
    model_cnn.add(Flatten())
    model_cnn.add(Dense(10, activation='linear'))
    model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(Dense(1))

    adm = keras.optimizers.Adam(learning_rate=0.0001)
    # Model Compiling (https://keras.io/models/sequential/)
    model_cnn.compile(loss='mse', optimizer=adm)
    #
    # # list some properties
    model_cnn.summary()
    return model_cnn


itrait = 0
trait_loss = []
trait_corr = []
for itrait in range(2, 3):
    # load Processed Data
    chunk_filelist = natsorted(glob.glob("../Data_Prediction/Compress_2_Chunk/" + "*.csv"))
    # chunk_filelist = chunk_filelist[0:1]
    # [0:1] -> It will run 1st csv file;
    # If you want to run 10000 files at a time, set [0:10000]
    print(chunk_filelist)
    index = 0
    chunk_corr = []
    chunk_loss = []
    for chunk_filename in chunk_filelist:
        X = pd.read_csv(chunk_filename, low_memory=False, index_col=0)
        X = X.T
        X.columns = np.arange(0, len(X.columns))
        X = X.drop(X.index.difference(Y.index))
        Y = Y.drop(Y.index.difference(X.index))
        missing_values_X = X.isnull().sum().sum()
        print("Total Missing values in X:", missing_values_X)
        print("X shape", itrait, index, X.shape)
        print("Y shape", itrait, index, Y.shape)

        X_train, X_valid, y_train, y_valid = train_test_split(X, Y[itrait], test_size=0.2)
        print("X_train shape and y_train shape", X_train.shape, y_train.shape)
        print("X_valid shape, y_valid shape", X_valid.shape, y_valid.shape)

        batch_size = 64
        epochs = 40
        nSNP = X_train.shape[1]
        print("SNP", nSNP)

        X2_train = np.expand_dims(X_train, axis=2)
        X2_valid = np.expand_dims(X_valid, axis=2)

        print("X2_train shape", X2_train.shape)
        print("X2_valid shape", X2_valid.shape)

        model_cnn = CNN(nSNP)
        plot_model(model_cnn, to_file='../ModelMetaData_Prediction/Net_2_CNN_Chunk/'
                    'Net_2_CNN_Chunk_Flowgraph/Net_2_CNN_Chunk_Flowgraph.png',
                   show_shapes=True, show_layer_names=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        # # training
        model_cnn_train = model_cnn.fit(X2_train, y_train, epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(X2_valid, y_valid),
                                        shuffle=True, callbacks=[es])

        # print(model_cnn_train.history)
        loss = model_cnn_train.history['loss']
        val_loss = model_cnn_train.history['val_loss']

        plt.plot(loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.title('Training and validation loss')
        plt.savefig('../ModelMetaData_Prediction/Net_2_CNN_Chunk/'
                    'Net_2_CNN_Chunk_Loss/'
                    'Net_2_CNN_Loss_Trait_{}'.format(itrait) +
                    '_chunk_{}'.format(index) + '.png')
        # plt.show()
        plt.clf()

        # cross-validation
        mse_prediction = model_cnn.evaluate(X2_valid, y_valid, batch_size=batch_size)
        print('\nMSE in prediction =', itrait, index, mse_prediction)
        chunk_loss.append(mse_prediction)

        # get predicted target values
        y_hat = model_cnn.predict(X2_valid, batch_size=batch_size)
        np.seterr(divide='ignore', invalid='ignore')

        # plot observed vs. predicted targets
        plt.title('CNN: Observed vs Predicted Y')
        plt.ylabel('Predicted')
        plt.xlabel('Observed')
        plt.scatter(y_valid, y_hat, marker='o')
        # obtain m (slope) and b(intercept) of linear regression line
        m, b = np.polyfit(y_valid, y_hat, 1)

        # add linear regression line to scatterplot
        plt.plot(y_valid, m * y_valid + b)
        plt.savefig('../ModelMetaData_Prediction/Net_2_CNN_Chunk/'
                    'Net_2_CNN_Chunk_Prediction/'
                    'Net_2_CNN_Prediction_Trait_{}'.format(itrait)
                    + '_chunk_{}'.format(index) + '.png')
        # plt.show()
        plt.clf()

        # correlation btw predicted and observed
        corr = np.corrcoef(y_valid, y_hat[:, 0])[0, 1]
        print('\nCorr obs vs pred =', itrait, index, corr)
        chunk_corr.append(corr)
        index += 1
    print("Chunk Loss: ", chunk_loss)
    print("Chunk Corr: ", chunk_corr)
    trait_loss.append([itrait, chunk_loss])
    trait_corr.append([itrait, chunk_corr])
    itrait += 1

# the star is unpacking the sublist to make a flat list
trait_loss_df = [[ele[0], *ele[1]] for ele in trait_loss]
trait_loss_df = pd.DataFrame(trait_loss_df,
                columns=['trait_id', 'chunk_0', 'chunk_1'])
# trait_loss_df.to_csv('../ModelMetaData_Prediction/Net_2_CNN_Chunk/'
#                      'Net_2_CNN_Chunk_Result/Net_2_CNN_Chunk_loss.csv')

# the star is unpacking the sublist to make a flat list
trait_corr_df = [[ele[0], *ele[1]] for ele in trait_corr]
trait_corr_df = pd.DataFrame(trait_corr_df,
                columns=['trait_id', 'chunk_0', 'chunk_1'])
# trait_corr_df.to_csv('../ModelMetaData_Prediction/Net_2_CNN_Chunk/'
#                      'Net_2_CNN_Chunk_Result/Net_2_CNN_Chunk_cor.csv')
#
# pathlib.Path("../ModelMetaData_Prediction/Net_2_CNN_Chunk/Net_2_CNN_Chunk_Time/"
#              "Net_2_CNN_Chunk_Training_Time.txt"). \
#     write_text("Net_2_CNN_Chunk_Training_Time: {}"
#                .format(time.time() - start_time))

print("Trait Loss: ", trait_loss_df)
print("Trait Corr: ", trait_corr_df)
print('Total Training Time: ', time.time() - start_time)

