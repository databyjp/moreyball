# ========== (c) JP Hwang 1/11/20  ==========

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
from viz import plot_shot_chart

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

shots_df = pd.read_csv('procdata/shots_df.csv', index_col=0)
all_teams_df = pd.read_csv('procdata/shot_dist_df.csv', index_col=0)

tmp_df = shots_df[shots_df["team"] == "PHI"]
tmp_df = tmp_df[tmp_df["on_court"].str.contains("Ben Simmons")]
tmp_df = tmp_df[tmp_df["player"] == "Joel Embiid"]
grp_df = tmp_df.groupby(["simple_zone", "shot_made"]).count().reset_index()[["simple_zone", "shot_made", "game_id"]]
# grp_df.to_csv("temp/embiid_w_simmons.csv")

# fig = plot_shot_chart(tmp_df, 'PHI', "All", "pps_abs", title="Shot chart")
# fig.show()

tmp_df = shots_df[shots_df["team"] == "PHI"]
tmp_df = tmp_df[-tmp_df["on_court"].str.contains("Ben Simmons")]
tmp_df = tmp_df[tmp_df["player"] == "Joel Embiid"]
grp_df = tmp_df.groupby(["simple_zone", "shot_made"]).count().reset_index()[["simple_zone", "shot_made", "game_id"]]
# grp_df.to_csv("temp/embiid_wo_simmons.csv")

fig = plot_shot_chart(shots_df, 'PHI', "All", "pps_abs", title="Shot chart")
fig.show()
