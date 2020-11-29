
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

    in_df.loc[in_df.rbin >= 31.5, "rbin"] = 31.5

    makes = in_df.groupby("rbin").shot_made.sum()
    shots = in_df.groupby("rbin").shot_made.count()
    acc = makes / shots

    grp_df = pd.concat([makes, shots, acc], axis=1)
    grp_df.columns = ['makes', 'shots', 'acc']
    grp_df.reset_index(inplace=True)

    grp_df = grp_df.assign(freq=np.round(100 * grp_df.shots / grp_df.shots.sum(), 2))
    grp_df = grp_df.assign(pps=np.round(grp_df.acc * 2 * 100, 2))
    grp_df.loc[grp_df.rbin >= 23.5, "pps"] = np.round(grp_df.loc[grp_df.rbin >= 23.5, "pps"] * 1.5, 2)

    grp_df = grp_df.assign(bin_name="TBD")
    grp_df.loc[(grp_df.rbin < 22.5), "bin_name"] = grp_df[(grp_df.rbin < 22.5)].rbin.apply(lambda x: str(int(x-1.5)) + " to " + str(int(x+1.5)))
    grp_df.loc[(grp_df.rbin == 22.5), "bin_name"] = "21+ (2pt)"
    grp_df.loc[(grp_df.rbin == 23.5), "bin_name"] = "22 to 23.75 (Cnr 3)"
    grp_df.loc[(grp_df.rbin == 25.5), "bin_name"] = "23.75 to 27"
    grp_df.loc[(grp_df.rbin > 25.5), "bin_name"] = grp_df[(grp_df.rbin > 23.5)].rbin.apply(lambda x: str(int(x-1.5)) + " to " + str(int(x+1.5)))
    grp_df.loc[(grp_df.rbin == 31.5), "bin_name"] = "30+"

    labels = {"pps": "PTS/100", "bin_name": "Distance (ft)", "freq": "Frequency (%)"}

    marker_cmin = 65
    marker_cmax = 145
    fig = px.bar(grp_df, x="bin_name", y="freq", range_y=[0, 40],
                 title=title_txt, labels=labels, category_orders={"bin_name": grp_df["bin_name"].values},
                 color="pps", color_continuous_scale=px.colors.diverging.RdYlBu_r, range_color=[marker_cmin, marker_cmax],
                 width=900, height=600,
                 template="plotly_white")
    fig.update_traces(marker_line_width=0.6, marker_line_color='navy')
    fig.update_layout(bargap=0.4)

    paper_bgcolor = "wheat"
    plot_bgcolor = "Cornsilk"

    fig.update_layout(
        margin=dict(l=60, r=20, t=50, b=20),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
    )

    ticktexts = [str(marker_cmin) + '-', "", str(marker_cmax) + '+']
    fig.update_layout(coloraxis_colorbar=dict(
        x=0.9,
        y=0.85,
        len=0.3,
        yanchor='middle',
        thickness=16,
        tickvals=[marker_cmin, (marker_cmin + marker_cmax) / 2, marker_cmax],
        ticktext=ticktexts,
        titlefont=dict(size=12),
        tickfont=dict(size=12),
    ))
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
    shots_df = dataproc.add_polar_bins(shots_df, rbin_size=30, large_tbins=True)

    filt_df = shots_df
    title_txt = f"NBA shot profile: '" + yr_a + "/'" + yr_b + " season"
    fig = build_shot_dist_bin_chart(filt_df, title_txt)
    fig.write_image(f"temp/by_dist_nba_{yr_a}_{yr_b}.png")
    fig.show()

    # teams = ["LAL", "MIA", "HOU", "TOR", "GSW", "NYK", "SAS", "CHA"]
    # for team in teams:
    #     filt_df = shots_df[shots_df.team == team]
    #     title_txt = f"{team} shot profile: '" + yr_a + "/'" + yr_b + " season"
    #     fig = build_shot_dist_bin_chart(filt_df, title_txt)
    #     fig.write_image(f"temp/by_dist_{team}_{yr_a}_{yr_b}.png")
    #     fig.show()
    #
    # players = ["Trae Young", "Damian Lillard", "Luka Doncic", "James Harden"]
    # for player in players:
    #     filt_df = shots_df[shots_df.player == player]
    #     title_txt = f"{player} shot profile: '" + yr_a + "/'" + yr_b + " season"
    #     fig = build_shot_dist_bin_chart(filt_df, title_txt)
    #     fig.write_image(f"temp/by_dist_{player}_{yr_a}_{yr_b}.png")
    #     fig.show()

    tm_pls = [("UTA", "Gobert"), ("PHI", "Joel Embiid"), ("LAL", "Anthony Davis"), ("GSW", "Draymond Green"), ("CLE", "Kevin Love")]
    for (team, player) in tm_pls:
        team_games = shots_df[shots_df.team == team].game_id.unique()
        team_games_df = shots_df[shots_df.game_id.isin(team_games)]
        opp_df = team_games_df[team_games_df.team != team]

        for on_state in [1, -1]:
            if on_state == 1:
                filt_df = opp_df[opp_df.on_court.notna()]
                filt_df = filt_df[filt_df.on_court.str.contains(player)]
                on_txt = "with"
            else:
                filt_df = opp_df[opp_df.on_court.notna()]
                filt_df = filt_df[-filt_df.on_court.str.contains(player)]
                on_txt = "without"

            title_txt = f"{team}: Opponent shots {on_txt} {player} on: '" + yr_a + "/'" + yr_b + " season"
            fig = build_shot_dist_bin_chart(filt_df, title_txt)
            fig.write_image(f"temp/by_dist_{on_txt}_{player}_{yr_a}_{yr_b}.png")
            fig.show()
