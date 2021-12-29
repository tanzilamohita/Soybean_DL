# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 15.11.2021
# =========================================

from sklearn.model_selection import train_test_split
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
from sklearn.metrics import mean_squared_error
import time
import pathlib
from numpy import mean

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
# using GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # model will be trained on GPU 0, if it is -1, GPU will not use for training

# load Compressed Data
compressed_fileList = natsorted(glob.glob("../ModelMetaData/Soybean_Net_1/Soybean_Net_1_EncData/"+"*.csv"))
compressed_fileList = compressed_fileList[160000:176920]  #change
print("Number of CSV Files: ", len(compressed_fileList))
print(compressed_fileList)

# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


# How many elements each list should have
n = 10
x = list(divide_chunks(compressed_fileList, n))
print(len(x))
print(x)

evaluate = []
n_rep = 20

# for rep in range(n_rep):
#     print("Repetition number: ", rep)
index = 16000   #change
mse_loss = []
start_time = time.time()
for i in range(len(x)):
    # print('Splitting input layer data: ', index, x[i])
    dfList = []

    for filename in x[i]:
        df = pd.read_csv(filename, header=None, low_memory=False)
        dfList.append(df)

    concatDf = pd.concat(dfList, axis=1)
    # print(concatDf)
    # export to csv
    # concatDf.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')
    data = np.array(concatDf)
    print("Shape of Data: ", i, data.shape)

    # Creating a training set, test set and validation set
    x_train, x_mid = train_test_split(data, test_size=0.4)
    x_test, x_valid = train_test_split(x_mid, test_size=0.5)
    print('Shape of train data: ', x_train.shape)
    print('Shape of test data: ', x_test.shape)
    print('Shape of validation data: ', x_valid.shape)

    # Taking the input data of dimension 30 and convert it to keras tensors.
    input = Input(shape=(data.shape[1],))
    print('Shape of input layer data: ', index, input.shape)

    # For all the hidden layers for the encoder and decoder
    # we use relu activation function for non-linearity.
    encoded = Dense(15, activation='relu')(input)
    encoded = Dense(5, activation='sigmoid')(encoded)

    decoded = Dense(15, activation='relu')(encoded)
    # The output layer needs to predict the probability of an output
    # which needs to either 0 or 1 and hence we use sigmoid activation function.
    decoded = Dense(data.shape[1], activation='sigmoid')(decoded)

    # this model maps an input to its reconstruction
    # creating the autoencoder with input data. Output will be the final decoder layer.
    autoencoder = Model(input, decoded)
    # extracting the encoder which takes input data and the output of encoder
    # is the encoded data of dimension 5
    encoder = Model(input, encoded)
    # the structure of the deep autoencoder model
    autoencoder.summary()
    # the structure of the encoder
    encoder.summary()
    # plot_model(encoder, to_file='../ModelMetaData/Soybean_Net_2/Soybean_Net_2_FlowGraph/'
    #                                 'Soybean_Net_2_EncFlowGraph_{}'.format(index) + '.png', show_shapes=True)
    # plot_model(autoencoder, to_file='../ModelMetaData/Soybean_Net_2/Soybean_Net_2_FlowGraph/'
    #                                'Soybean_Net_2_FlowGraph_{}'.format(index) + '.png', show_shapes=True)
    # # compiling the autoencoder model with adam optimizer.
    adm = tf.keras.optimizers.Adam(learning_rate=0.001)
    # As One Hot Encoded Data has a value of 0 0r 1 we use binary_crossentropy as the loss function
    # We use accuracy as the metrics used for the performance of the model.
    autoencoder.compile(optimizer=adm, loss='mse')
    # We finally train the autoencoder using the training data with 100 epochs and batch size of 52.
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    history = autoencoder.fit(x_train, x_train,
                              epochs=300,
                              batch_size=32,
                              shuffle=True,
                              validation_data=(x_valid, x_valid)
                              , callbacks=[es])

    # predicting the test using encoder
    enc_out = encoder.predict(data)
    # save encoded data into CSV file
    np.savetxt('../ModelMetaData/Soybean_Net_2/Soybean_Net_2_EncData/Soybean_Net_2_EncData_{}'.format(index)
               + '.csv', enc_out, delimiter=",", fmt='%1.0f')

    # reconstructing the input from autoencoder
    dec_out = autoencoder.predict(data)
    print('Shape of Decoded Data Output: ', dec_out.shape)
    print(dec_out)

    # save decoded data into CSV file
    np.savetxt('../ModelMetaData/Soybean_Net_2/Soybean_Net_2_DecData/Soybean_Net_2_DecData_{}'.format(index)
               + '.csv', dec_out, delimiter=",", fmt='%1.0f')
    # np.savetxt('Test/Analog_Output_fmt.csv', dec_out, delimiter=",", fmt='%1.0f')
    # save model
    filename = '../ModelMetaData/Soybean_Net_2/Soybean_Net_2_Model/Soybean_Net_2_Model_{}'.format(
        index) + '.h5'
    autoencoder.save(filename)
    score = autoencoder.evaluate(x_test, x_test, verbose=0)
    score_loss = score * 100

    # score_array = np.column_stack((score_loss, score_acc))
    evaluate.append([score_loss])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    result = np.array([loss, val_loss])
    # np.savetxt('../../../Data/HDRA/ModelMetadata/Net_1/Net_1_3/Net_1_3_Log/Net_1_3_Log_{}'.format(index) +
    #            '.csv', np.column_stack(np.array(result)), delimiter=',', fmt='%1.3f',
    #            comments='', header='Loss,Val_loss')

    print(history.history.keys())
    print('Loss: {:.3f}'.format(score_loss))

    mse = mean_squared_error(data, dec_out)
    print('SKlearn MSE: ', mse)
    mse_loss.append([mse])

    # plotting history for loss
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('../ModelMetaData/Soybean_Net_2/Soybean_Net_2_Plot/Soybean_Net_2_PlotLoss_{}'.format(index) + '.png')
    # plt.show()
    plt.clf()
    index += 1

#change
np.savetxt('../ModelMetaData/Soybean_Net_2/Soybean_Net_2_Evaluate/'
           'Soybean_Net_2_Evaluate_160000-176920.csv', evaluate, delimiter=',',
           fmt='%1.3f', comments='', header='Loss')
np.savetxt('../ModelMetaData/Soybean_Net_2/Soybean_Net_2_MSE/'
           'Soybean_Net_2_MSE_160000-176920.csv', mse_loss, delimiter=',',
           fmt='%1.5f', comments='', header='MSE Loss')

pathlib.Path("../ModelMetaData/Soybean_Net_2/Soybean_Net_2_Time/"
        "Soybean_Net_2_TrainingTime_160000-176920.txt")\
        .write_text("HDRA_Net_2_Training_Time: {}"
        .format(time.time() - start_time))
pathlib.Path("../ModelMetaData/Soybean_Net_2/Soybean_Net_2_MSE/"
        "Soybean_Net_2_MSE_160000-176920.txt").write_text("Soybean Net_2 Mean MSE Loss: {}"
        .format(mean(mse_loss)))

print("MSE Loss: ", mean(mse_loss))
print('Total Training Time: ', time.time() - start_time)
