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
Net_2_Compressed_fileList = natsorted(glob.glob("../ModelMetaData/Soybean_Net_2/"
                            "Soybean_Net_2_EncData/"+"*.csv"))
# Net_2_Compressed_fileList = Net_1_Compressed_fileList[0:5]
Net_2_Compressed_csv = pd.concat([pd.read_csv(f, header=None) for f in Net_2_Compressed_fileList], axis=1)
# Net_2_Compressed_csv = Net_2_Compressed_csv.transpose()
print(Net_2_Compressed_csv.shape)

Net_2_Compressed_csv.to_csv("../Data/Soybean_CompressedData/Soybean_Compress_2.csv",
                       index=False, header=None)




