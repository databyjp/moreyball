# ========== (c) JP Hwang 27/11/20  ==========

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

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)


# ==========  Draw similar bits in angular coordinates ==========
yrs = list(range(5, 20))

for yr in yrs[-1:]:

    # ===== Load DataFrame with shot data
    yr_a = ("0" + str(yr))[-2:]
    yr_b = ("0" + str(yr+1))[-2:]
    df_path = f"procdata/shots_df_{yr_a}_{yr_b}.csv"
    shots_df = pd.read_csv(df_path)
    shots_df = shots_df[shots_df.data_set.str.contains("Regular")]

    players = ["James Harden", "LeBron James", "Luka Doncic"]
    player = players[0]

    pl_df = shots_df[shots_df.player == player]

    # By game log
    pl_makes = pl_df.groupby("date").shot_made.sum()
    pl_shots = pl_df.groupby("date").shot_made.count()
    pl_pct = pl_makes/pl_shots
