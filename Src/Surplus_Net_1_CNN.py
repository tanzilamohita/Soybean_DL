# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/8/2022
# ===============================
import os
import pandas as pd
import numpy as np

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

df_column_name = pd.read_csv('../Data_Prediction/Soybean_X_C3_Test.csv', index_col=0,
                    low_memory=False)
# print(df_column_name.columns)

df = pd.read_csv('../Data_Prediction/Soybean_X_C1.csv', index_col=0,
                    low_memory=False, chunksize=58973)
category_number = 9
categories = [[] for i in range(category_number)]
index = 0
for x in df:
    # print(x.shape)
    x = np.array(x)
    for row in x:
        # print(row[0])
        categories[index].append(row)
        index = (index+1) % category_number

chunk_num = 0
for chunk in categories:
    chunk = pd.DataFrame(chunk)
    print(chunk.shape)
    chunk.to_csv('../Data_Prediction/Compress_1_Chunk/'
                 'Soybean_X_C1_{}'.format(chunk_num)
                 + '.csv', header=df_column_name.columns)
    chunk_num += 1

