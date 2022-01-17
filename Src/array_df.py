# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/14/2022
# ===============================

import numpy as np
import pandas as pd
trait_corr = [[0, [0.13, 0.24, 0.19]],
              [1, [0.10, 0.21, 0.06]],
              [2, [0.40, 0.29, 0.25]]]

trait_corr = [[ele[0], *ele[1]] for ele in trait_corr]
# print(trait_corr)
trait_corr = pd.DataFrame(trait_corr,
                columns=['trait_id', 'chunk_0', 'chunk_1', 'chunk_2'])
print(trait_corr)
