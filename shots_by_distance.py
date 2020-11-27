
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
import dataproc
import plotly.express as px

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)


def build_shot_dist_bin_chart(in_df, title_txt):
    bin_max = 300
    bin_increment = 30
    dist_bins = list(range(bin_max, 0, -bin_increment))
    bin_max_name = str(int(max(dist_bins) / 10)) + "+ ft"
    in_df = in_df.assign(dist_bin=bin_max_name)
    bin_names = [bin_max_name]
    for d in dist_bins:
        bin_name = str(int((d - bin_increment) / 10)) + "-" + str(int(d / 10)) + " ft"
        in_df.loc[(in_df.shot_dist_calc <= d), "dist_bin"] = bin_name
        bin_names.append(bin_name)
    bin_names = bin_names[::-1]

    makes = in_df.groupby("dist_bin").shot_made.sum()
    shots = in_df.groupby("dist_bin").shot_made.count()
    pct = np.round(100 * makes / shots, 2)

    grp_df = pd.concat([makes, shots, pct], axis=1)
    grp_df.columns = ['makes', 'shots', 'pct']
    grp_df.reset_index(inplace=True)

    grp_df['dist_bin'] = pd.Categorical(grp_df['dist_bin'], bin_names)
    grp_df.sort_values('dist_bin', inplace=True)
    grp_df = grp_df.assign(freq=np.round(100 * grp_df.shots / grp_df.shots.sum(), 2))

    labels = {"pct": "Accuracy (%)", "dist_bin": "Distance", "freq": "Frequency (%)"}
    fig = px.bar(grp_df, x="dist_bin", y="freq", range_y=[0, 35],
                 title=title_txt, labels=labels,
                 color="pct", color_continuous_scale=px.colors.diverging.RdYlBu_r, range_color=[30, 70],
                 width=900, height=600,
                 category_orders={"dist_bin": bin_names}, template="plotly_white")
    fig.update_traces(marker=dict(line=dict(width=1, color='Navy')),
                      selector=dict(mode='markers'))
    fig.update_layout(bargap=0.4)
    return fig


# ==========  Draw similar bits in angular coordinates ==========
yrs = list(range(5, 20))

for yr in yrs[-1:]:

    # ===== Load DataFrame with shot data
    yr_a = ("0" + str(yr))[-2:]
    yr_b = ("0" + str(yr+1))[-2:]
    df_path = f"procdata/shots_df_{yr_a}_{yr_b}.csv"
    shots_df = pd.read_csv(df_path)
    shots_df = shots_df[shots_df.data_set.str.contains("Regular")]
    shots_df = dataproc.add_polar_columns(shots_df)

    filt_df = shots_df
    title_txt = f"NBA shot profile: '" + yr_a + "/'" + yr_b + " season<BR>Color: Accuracy"
    fig = build_shot_dist_bin_chart(filt_df)
    fig.write_image(f"temp/by_dist_nba_{yr_a}_{yr_b}.png")
    fig.show()

    teams = ["LAL", "MIA", "HOU", "TOR", "GSW", "NYK", "SAS", "CHA"]
    for team in teams:
        filt_df = shots_df[shots_df.team == team]
        title_txt = f"{team} shot profile: '" + yr_a + "/'" + yr_b + " season<BR>Color: Accuracy"
        fig = build_shot_dist_bin_chart(filt_df, title_txt)
        fig.write_image(f"temp/by_dist_{team}_{yr_a}_{yr_b}.png")
        fig.show()

    players = ["Trae Young", "Damian Lillard", "Luka Doncic", "James Harden"]
    for player in players:
        filt_df = shots_df[shots_df.player == player]
        title_txt = f"{player} shot profile: '" + yr_a + "/'" + yr_b + " season<BR>Color: Accuracy"
        fig = build_shot_dist_bin_chart(filt_df, title_txt)
        fig.write_image(f"temp/by_dist_{player}_{yr_a}_{yr_b}.png")
        fig.show()