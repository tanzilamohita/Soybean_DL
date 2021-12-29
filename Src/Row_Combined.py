# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 12/24/2021
# ===============================

import pandas as pd

X = pd.read_csv('../Data_Prediction/Soybean_Compress_1.csv', header=None)#.iloc[0:, 1:]
# X.columns = np.arange(0, len(X.columns))
print(X.shape)
row_extract = pd.read_csv('../Data_Prediction/Soybean_Row_Extract.csv', header=None)
row_extract = row_extract[0].astype(str).tolist()
print(row_extract)
#
X = X.T
# X.columns = row_extract
X.to_csv('../Data_Prediction/Soybean_X_C1.csv', header=row_extract)