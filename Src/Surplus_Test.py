# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/5/2022
# ===============================

import os
import pandas as pd
import numpy as np

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

# shuffle_chunk = chunk.iloc[np.random.permutation(len(chunk))]
# id = chunk.columns
# chunk[id] = chunk[id].sample(frac=1).to_numpy()
# print(chunk)
# print(chunk.index%3)

Y = pd.read_csv('../Data_Prediction/Soybean_Y.csv', index_col=0)
missing_values_Y = Y.isnull().sum().sum()
print("Total Missing values in Y:", missing_values_Y)
Y = Y.dropna()
Y.columns = np.arange(0, len(Y.columns))

df_column_name = pd.read_csv('../Data_Prediction/Soybean_X_C3_Test.csv', index_col=0,
                    low_memory=False)
print(df_column_name.columns)

df = pd.read_csv('../Data_Prediction/Soybean_X_C3.csv', index_col=0,
                    low_memory=False, chunksize=5000)
category_number = 3
categories = [[] for i in range(category_number)]
# chunk1 = []
# chunk2 = []
# chunk3 = []
index = 0
for x in df:
    # print(x.shape)
    x = np.array(x)
    for row in x:
        # print(row[0])
        categories[index].append(row)
        index = (index+1)%category_number


# for chunk in categories:
#     print(len(chunk))

df_0 = pd.DataFrame(categories[0])
print(df_0)
df_0.to_csv('../Data_Prediction/Soybean_X_C3_test_chunk_0.csv', header=df_column_name.columns)
