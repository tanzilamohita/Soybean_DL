# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 12/31/2021
# ===============================
# main modules needed
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
import tensorflow
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
print("Y shape", Y.shape)

chunk = pd.read_csv('../Data_Prediction/Soybean_X_C3.csv', index_col=0,
                low_memory=False, chunksize=5000)
chunk_num = 0
chunk_loss = []
chunk_val_loss = []
chunk_cor = []
chunk_y_valid = []
chunk_y_hat = []
chunk_mse_prediction = []

for X in chunk:
    X = X.T
    X.columns = np.arange(0, len(X.columns))
    X = X.drop(X.index.difference(Y.index))
    Y = Y.drop(Y.index.difference(X.index))
    missing_values_X = X.isnull().sum().sum()
    print("Total Missing values in X:", missing_values_X)
    # print("X shape", chunk_num, X.shape)
    # print("Y shape", chunk_num, Y.shape)

    # data partitioning into train and validation
    index = 0  # first trait analyzed
    corr_df = []
    # print(Y[itrait])
    # Y.shape[1]
    for i in range(0, 3):
        print(Y[i])
        print(i)
        print("X shape", chunk_num, i, X.shape)
        print("Y shape", chunk_num, i, Y.shape)
        X_train, X_valid, y_train, y_valid = train_test_split(X, Y[i], test_size=0.2)
        print("X_train shape and y_train shape", X_train.shape, y_train.shape)
        print("X_valid shape, y_valid shape", X_valid.shape, y_valid.shape)
        i += 1
    #     batch_size = 128
    #     epochs = 1
    #     nSNP = X_train.shape[1]
    #     print("SNP", nSNP)
    #     nStride = 2  # stride between convolutions
    #
    #     X2_train = np.expand_dims(X_train, axis=2)
    #     X2_valid = np.expand_dims(X_valid, axis=2)
    #
    #     print("X2_train shape", X2_train.shape)
    #     print("X2_valid shape", X2_valid.shape)
    #     # Instantiate
    #     model_cnn = Sequential()
    #     # add convolutional layer
    #     model_cnn.add(Conv1D(32, 2, activation="relu", input_shape=(nSNP, 1)))
    #     # add pooling layer: takes maximum of two consecutive values
    #     model_cnn.add(MaxPooling1D(pool_size=2))
    #
    #     # # add convolutional layer
    #     model_cnn.add(Conv1D(64, 2, activation='relu'))
    #     # # add pooling layer: takes maximum of two consecutive values
    #     model_cnn.add(MaxPooling1D(pool_size=2))
    #     # # add convolutional layer
    #     model_cnn.add(Conv1D(128, 2, activation='relu'))
    #     # # add pooling layer: takes maximum of two consecutive values
    #     model_cnn.add(MaxPooling1D(pool_size=2))
    #
    #     # Solutions above are linearized to accommodate a standard layer
    #     model_cnn.add(Flatten())
    #     model_cnn.add(Dense(64, activation='linear'))
    #     # model_cnn.add(LeakyReLU(alpha=0.1))
    #     model_cnn.add(Dense(1))
    #
    #     adm = keras.optimizers.Adam(learning_rate=0.001)
    #     # Model Compiling (https://keras.io/models/sequential/)
    #     model_cnn.compile(loss='mse', optimizer=adm)
    #     #
    #     # # list some properties
    #     model_cnn.summary()
    #
    #     # plot_model(model_cnn, to_file='../ModelMetaData_Prediction/Net_3_CNN_Chunk/'
    #     #                         'Net_3_CNN_Chunk_Flowgraph/Net_3_CNN_Chunk_FlowgraphTrait_{}'.format(i)
    #     #             + '_chunk_num_{}'.format(chunk_num) + '.png',
    #     #            show_shapes=True, show_layer_names=True)
    #
    #     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    #     # # training
    #     model_cnn_train = model_cnn.fit(X2_train, y_train, epochs=epochs,
    #                                     batch_size=batch_size,
    #                                     validation_data=(X2_valid, y_valid),
    #                                     shuffle=True, callbacks=[es])
    #
    #     # print(model_cnn_train.history)
    #     loss = model_cnn_train.history['loss']
    #     val_loss = model_cnn_train.history['val_loss']
    #
    #     chunk_loss.append(loss)
    #     chunk_val_loss.append(val_loss)
    #
    #     # np.savetxt('../ModelMetaData_Prediction/Net_3_CNN_Chunk/Net_3_CNN_Chunk_Result/'
    #     #            'Net_3_CNN_Chunk_Loss/Net_3_CNN_Chunk_loss_Trait_{}'.format(i)
    #     #            + '_chunk_num_{}'.format(chunk_num) + '.csv',
    #     #            loss, delimiter=',', comments='')
    #     # np.savetxt('../ModelMetaData_Prediction/Net_3_CNN_Chunk/Net_3_CNN_Chunk_Result/'
    #     #            'Net_3_CNN_Chunk_Loss/Net_3_CNN_Chunk_val_loss_Trait_{}'.format(i)
    #     #            + '_chunk_num_{}'.format(chunk_num) + '.csv',
    #     #            val_loss, delimiter=',', comments='')
    #
    #     # cross-validation
    #     chunk_mse_prediction.append(model_cnn.evaluate(X2_valid, y_valid, batch_size=batch_size))
    #
    #
    #     # get predicted target values
    #     y_hat = model_cnn.predict(X2_valid, batch_size=batch_size)
    #     np.seterr(divide='ignore', invalid='ignore')
    #
    #     corr = np.corrcoef(y_valid, y_hat[:, 0])[0, 1]
    #     print("y_hat", y_hat)
    #     print('\nCorr obs vs pred =', corr)
    #     chunk_y_valid = np.append(chunk_y_valid, y_valid)
    #     chunk_y_hat = np.append(chunk_y_hat, y_hat)
    #
    #     # np.savetxt('../ModelMetaData_Prediction/Net_3_CNN_Chunk/Net_3_CNN_Chunk_Result/'
    #     #            'Net_3_CNN_Chunk_Prediction/Net_3_CNN_Chunk_Y_hat_Trait_{}'.format(i)
    #     #             + '_chunk_num_{}'.format(chunk_num) + '.csv',
    #     #            y_hat, delimiter=',', comments='')
    #     # np.savetxt('../ModelMetaData_Prediction/Net_3_CNN_Chunk/Net_3_CNN_Chunk_Result/'
    #     #            'Net_3_CNN_Chunk_Prediction/Net_3_CNN_Chunk_Y_valid_Trait_{}'.format(i)
    #     #             + '_chunk_num_{}'.format(chunk_num) + '.csv',
    #     #            y_valid, delimiter=',', comments='')
    #     index += 1
    # # plt.plot(chunk_loss, label='Training loss')
    # # plt.plot(chunk_val_loss, label='Validation loss')
    # # plt.ylabel('loss')
    # # plt.xlabel('epoch')
    # # plt.legend(['train', 'valid'], loc='upper left')
    # # plt.title('Training and validation loss')
    # # # plt.savefig('../ModelMetaData_Prediction/Net_3_CNN_Chunk/Net_3_CNN_Chunk_Loss/'
    # # #             'Net_3_CNN_Chunk_Loss_Trait_{}'.format(i)
    # # #             + '_chunk_num_{}'.format(chunk_num) + '.png')
    # # plt.show()
    # # plt.clf()
    # #
    # # print('\nMSE in prediction =', chunk_mse_prediction)
    # # # correlation btw predicted and observed
    # # corr = np.corrcoef(chunk_y_valid, chunk_y_hat[:, 0])[0, 1]
    # # print('\nCorr obs vs pred =', corr)
    #
    chunk_num += 1

# # plot observed vs. predicted targets
# plt.title('CNN: Observed vs Predicted Y')
# plt.ylabel('Predicted')
# plt.xlabel('Observed')
# plt.scatter(chunk_y_valid, chunk_y_hat, marker='o')
# # obtain m (slope) and b(intercept) of linear regression line
# m, b = np.polyfit(chunk_y_valid, chunk_y_hat, 1)
#
# # add linear regression line to scatterplot
# plt.plot(chunk_y_valid, m * chunk_y_valid + b)
# # plt.savefig('../ModelMetaData_Prediction/Net_3_CNN_Chunk/Net_3_CNN_Chunk_Prediction/'
# #             'Net_3_CNN_Chunk_Prediction_Trait_{}'.format(i)
# #             + '_chunk_num_{}'.format(chunk_num) + '.png')
# plt.show()
# plt.clf()
# print('chunk_y_hat', chunk_y_hat)


# np.savetxt('../ModelMetaData_Prediction/Net_3_CNN_Chunk/Net_3_CNN_Chunk_Result/'
#            'Net_3_CNN_Chunk_Prediction/Net_3_CNN_Chunk_cor.csv',
#            np.column_stack(chunk_cor), delimiter=',', comments='', fmt='%s')
#
# pathlib.Path("../ModelMetaData_Prediction/Net_3_CNN_Chunk/Net_3_CNN_Chunk_Time/"
#              "Net_3_CNN_Chunk_Training_Time.txt"). \
#     write_text("Training Time for Raw: {}"
#                .format(time.time() - start_time))
print('Total Training Time: ', time.time() - start_time)



