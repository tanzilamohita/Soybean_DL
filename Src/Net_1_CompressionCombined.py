# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 12/17/2021
# ===============================
import pandas as pd
import numpy as np
from natsort import natsorted
import glob


# load Compressed Data
Net_1_Compressed_fileList = natsorted(glob.glob("../ModelMetaData/Soybean_Net_1/"
                            "Soybean_Net_1_EncData/"+"*.csv"))
# Net_1_Compressed_fileList = Net_1_Compressed_fileList[0:5]
Net_1_Compressed_csv = pd.concat([pd.read_csv(f, header=None) for f in Net_1_Compressed_fileList], axis=1)
# Net_1_Compressed_csv = Net_1_Compressed_csv.transpose()
print(Net_1_Compressed_csv.shape)

Net_1_Compressed_csv.to_csv("../Data/Soybean_CompressedData/Soybean_Compress_1.csv",
                       index=False, header=None)




