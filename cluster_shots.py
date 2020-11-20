# ========== (c) JP Hwang 20/11/20  ==========

import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

df_path = "procdata/shots_df_19_20.csv"
shots_df = pd.read_csv(df_path)

shots_df

X = np.array([[1, 2], [2, 2], [2, 3],
              [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
