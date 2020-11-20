# ========== (c) JP Hwang 17/11/20  ==========

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
import viz

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

yrs = list(range(5, 20))

stats_list = list()
grp_df_list = list()
grp_make_df_list = list()

for yr in yrs[:]:

    yr_a = ("0" + str(yr))[-2:]
    yr_b = ("0" + str(yr+1))[-2:]
    shots_df = pd.read_csv(f'procdata/shots_df_{yr_a}_{yr_b}.csv', index_col=0)

    threes_df = shots_df[shots_df["is_three"]]
    twos_df = shots_df[-shots_df["is_three"]]

    three_freq = len(threes_df) / len(shots_df)

    shot_dist_mean = shots_df["shot_distance"].mean()
    shot_dist_three_mean = threes_df["shot_distance"].mean()
    shot_dist_two_mean = twos_df["shot_distance"].mean()

    shot_acc = sum(shots_df["shot_made"]) / len(shots_df)
    shot_acc_three = sum(threes_df["shot_made"]) / len(threes_df["shot_made"])
    shot_acc_two = sum(twos_df["shot_made"]) / len(twos_df["shot_made"])

    grp_df = shots_df.groupby("simple_zone").count()["game_id"].reset_index()
    # grp_df = grp_df.assign(pts=0)

    grp_make_df = shots_df[shots_df.shot_made == 1].groupby("simple_zone").count()["game_id"].reset_index()
    # grp_make_df = grp_make_df.assign(pts=0)

    # for i, row in grp_make_df.iterrows():
    #     if i <= 2:
    #         grp_make_df.loc[i, "pts"] = row["game_id"] * 2
    #     else:
    #         grp_make_df.loc[i, "pts"] = row["game_id"] * 3
    # grp_make_df = grp_make_df.assign(pts_pct=grp_make_df.pts/grp_make_df.pts.sum())

    data_dict = {"Season": "'" + yr_a + "-'" + yr_b,
                 "Threes Freq": three_freq,
                 "Twos Freq": 1-three_freq,

                 "Avg shot dist": shot_dist_mean,
                 "Avg shot dist - 2": shot_dist_two_mean,
                 "Avg shot dist - 3": shot_dist_three_mean,

                 "Overall acc": shot_acc,
                 "Two acc": shot_acc_two,
                 "Three acc": shot_acc_three
                 }

    grp_df = grp_df.assign(season="'" + yr_a + "-'" + yr_b)
    grp_df_list.append(grp_df)

    grp_make_df = grp_make_df.assign(season="'" + yr_a + "-'" + yr_b)
    grp_make_df_list.append(grp_make_df)

    stats_list.append(data_dict)

tot_grp_make_df = pd.concat(grp_make_df_list)
tot_grp_df = pd.concat(grp_df_list)

tot_grp_df = tot_grp_df.assign(acc=tot_grp_make_df.game_id/tot_grp_df.game_id)


tot_grp_df = tot_grp_df[["simple_zone", "acc", "season"]].pivot(columns=['simple_zone'], index=["season"]).reset_index()
tot_grp_df.to_csv("temp/acc_grp_df.csv")

df = pd.DataFrame(stats_list)
df.to_csv("temp/compiled_data.csv")

# df_melt = df.melt(id_vars=["Season"], value_vars=["Twos Freq", "Threes Freq"])
#
# import plotly.express as px
# fig = px.bar(df_melt, x="Season", y="value", color="variable", barmode="stack",
#              template="plotly_white")
# fig.show()

