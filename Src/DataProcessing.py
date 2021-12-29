# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 08.10.2021
# =========================================

import numpy as np
import pandas as pd
from keras import Input
import os
import time

# using GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # model will be trained on GPU 0, if it is -1, GPU will not use for training

start_time = time.time()
print("Loading Data")
# loading data
def Data():
    df = pd.read_csv('../Data/Raw/Soybean_Data.csv', low_memory=False, na_values=["N"])
    df_transpose = df.transpose()
    df_range = df_transpose.iloc[1:, 0:]
    gen_data = np.array(df_range)
    return gen_data.T


print("The shape of data: ", Data().shape)
# print(Data())


# print(HDRA_GenData())
# missing_values = HDRA_GenData().isnull().sum().sum()
# print("Total Missing values:", missing_values)


# splitting input data
onehotinp = np.hsplit(Data(), 176919)
split_input = np.array(onehotinp)
print(split_input[0])
print("Splitting Input data: ", split_input[0].shape)

index = 0
for i in range(len(split_input)):
    # print('Splitting input layer data: ', index, split_input[i].shape)
    # deep autoencoder
    input = Input(shape=(split_input[i].shape[1],))
    print('Shape of input layer data: ', index, input.shape)
    np.savetxt('../Data/Split_Input_Data/Split_Input_Soybean_Data_{}'.format(index)
               + '.csv', split_input[i], delimiter=",", fmt='%1.0f')
    index += 1

print("Time:", (time.time() - start_time))
print("Completed")

