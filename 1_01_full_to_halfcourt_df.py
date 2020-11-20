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
from dataproc import to_halfcourt_df
import os

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

src_dir = "procdata/old"
out_dir = "procdata"
df_files = [f for f in os.listdir(src_dir) if "shots_df_" in f]

for df_file in df_files:
    logger.info(f"Loading {df_file}")
    src_path = os.path.join(src_dir, df_file)
    shots_df = pd.read_csv(src_path, index_col=0)

    logger.info(f"Processing {df_file}")
    shots_df = to_halfcourt_df(shots_df)

    logger.info(f"Saving {df_file}")
    out_path = os.path.join(out_dir, df_file)
    shots_df.to_csv(out_path)
