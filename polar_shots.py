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
import viz
import plotly.express as px
import plotly.graph_objects as go

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

fig_width = 600

labels = {
    "rel_pct": "Relative<BR>accuracy (%)", "pps": "PTS/100",
    "tbin": "Shot Angle (degrees)", "rbin": "Shot Distance (ft)"
}


def grp_polar_shots(shots_df, tbin_smoothing_bins=1):
    """
    :param shots_df:
    :param tbin_smoothing_bins: How many adjacent (anglular) bins to use for data smoothing
    :return:
    """
    grp_shots_df = shots_df.groupby(["tbin", "rbin"]).count()["game_id"]
    grp_makes_df = shots_df.groupby(["tbin", "rbin"]).sum()["shot_made"]
    grp_pcts_df = grp_makes_df / grp_shots_df

    grp_scores_df = shots_df.groupby(["tbin", "rbin", "is_three"]).sum()["shot_made"].reset_index()
    grp_scores_df = grp_scores_df.assign(points=0)
    grp_scores_df.loc[grp_scores_df["is_three"] == True, "points"] = grp_scores_df[grp_scores_df["is_three"] == True]["shot_made"] * 3
    grp_scores_df.loc[grp_scores_df["is_three"] == False, "points"] = grp_scores_df[grp_scores_df["is_three"] == False]["shot_made"] * 2
    grp_scores_df = grp_scores_df.groupby(["tbin", "rbin"]).sum()["points"]
    # No averaging - at the same distance
    grp_pps_df = grp_scores_df / grp_shots_df * 100

    grp_shots_df = grp_shots_df.reset_index()
    grp_shots_df = grp_shots_df.rename({"game_id": "attempts"}, axis=1)
    grp_shots_df = grp_shots_df.assign(pct=100 * grp_pcts_df.reset_index()[0])
    grp_shots_df = grp_shots_df.assign(pps=grp_pps_df.values)

    grp_shots_df = grp_shots_df.assign(rel_pct=0)
    for i, row in grp_shots_df.iterrows():
        avg = grp_shots_df[(np.abs(grp_shots_df["tbin"]) == np.abs(row["tbin"])) & (grp_shots_df["rbin"] == row["rbin"])].pct.mean()
        grp_shots_df.loc[i, "rel_pct"] = row["pct"] - avg
    grp_shots_df = grp_shots_df.assign(better_side=np.sign(grp_shots_df.rel_pct))

    # Perform averaging for PPS - keep distance constance, only average by adjacent angle bins
    tbin_thresh = tbin_smoothing_bins * abs(np.sort(shots_df.tbin.unique())[0] - np.sort(shots_df.tbin.unique())[1])
    for i, row in grp_shots_df.iterrows():
        temp_rows = grp_shots_df[
            (grp_shots_df["rbin"] == row["rbin"]) &
            (grp_shots_df["tbin"] <= row["tbin"] + tbin_thresh) &
            (grp_shots_df["tbin"] >= row["tbin"] - tbin_thresh)
        ]
        tot_pts = np.sum(temp_rows["pps"] * temp_rows["attempts"])  # Make sure to average using totals
        tot_shots = np.sum(temp_rows["attempts"])
        mean_pps = tot_pts/tot_shots
        grp_shots_df.loc[row.name, "pps"] = mean_pps

    for i, row in grp_shots_df.iterrows():
        min_samples = 0.0005
        if row["attempts"] < (grp_shots_df["attempts"].sum() * min_samples):
            grp_shots_df.loc[row.name, "attempts"] = 0

    return grp_shots_df


def format_polar_chart(fig, marker_cmax, marker_cmin):
    fig.update_traces(marker=dict(line=dict(width=0.6, color="#333333"), symbol="square"))
    paper_bgcolor = "wheat"
    plot_bgcolor = "Cornsilk"
    fig.update_traces(marker=dict(cmax=marker_cmax, cmin=marker_cmin))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            zeroline=True,
            showline=True,
            ticks='',
            showticklabels=True,
            fixedrange=True,
            zerolinewidth=0.2,
            zerolinecolor="#dddddd",
            tickfont={"size": 11},
            title_font={"size": 13},
        ),
        xaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            ticks='',
            showticklabels=True,
            fixedrange=True,
            zerolinewidth=0.2,
            zerolinecolor="#dddddd",
            tickfont={"size": 11},
            title_font={"size": 13},
        ),
    )
    ticktexts = [str(marker_cmin) + '-', "", str(marker_cmax) + '+']
    fig.update_layout(coloraxis_colorbar=dict(
        # thickness=15,
        x=0.88,
        y=0.83,
        thickness=15,
        yanchor='middle',
        len=0.3,
        tickvals=[marker_cmin, (marker_cmin + marker_cmax) / 2, marker_cmax],
        ticktext=ticktexts,
        titlefont=dict(size=12),
        tickfont=dict(
            size=12,
            color=textcolor
        )
    ))
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=45))
    return fig


def plot_polar_pps(tmp_df, marker_cmin=75, marker_cmax=125):

    fig = px.scatter(tmp_df, x="tbin", y="rbin", size="attempts", color="pps",
                     color_continuous_scale=px.colors.diverging.RdYlBu_r, template="plotly_white",
                     range_x=[-120, 120], range_y=[-4, 40], range_color=[marker_cmin, marker_cmax],
                     height=600, width=750, size_max=14, labels=labels
                     )
    fig = format_polar_chart(fig, marker_cmax=marker_cmax, marker_cmin=marker_cmin)
    fig = viz.add_jph_signature(fig, yloc=0.03)

    return fig


def add_polar_bins(shots_df, tbin_size=9, rbin_size=30):
    # Group shots into buckets / bins
    # One challenge is the pesky 3-point line - between 22 and 23.75 feet, some shots are threes and some aren't
    # So let's make sure that no groups have 3s and 2s coexist.
    # The 3pt arc meets the corner 3 line at 22.13 degrees

    shots_df = shots_df.assign(tbin=tbin_size * np.sign(shots_df.angle) * ((np.abs(shots_df.angle) + (tbin_size/2)) // tbin_size))
    shots_df = shots_df.assign(rbin=0.1 * rbin_size * (0.5 + (np.abs(shots_df.shot_dist_calc) // rbin_size)))

    # For the last bins of twos
    shots_df.loc[(shots_df.shot_dist_calc >= 210) & (shots_df.is_three == False), "rbin"] = 22.5
    shots_df.loc[(shots_df.shot_dist_calc >= 210) & (shots_df.angle < -67.5), "rbin"] = 19.5
    shots_df.loc[(shots_df.shot_dist_calc >= 210) & (shots_df.angle > 67.5), "rbin"] = 19.5

    # For bins of threes
    shots_df.loc[shots_df.is_three == True, "rbin"] = 24 + 0.1 * rbin_size * (0.5 + (np.abs(shots_df.shot_dist_calc-240) // rbin_size))

    # For corner threes:
    for temp_t in [67.5, 76.5, 85.5, 94.5]:
        if temp_t == 67.5:
            new_rbin = 24.5  # This is an awkward bin - between corner and the full length; so show it as such
        else:
            new_rbin = 23.5  # Corner 3
        # Left corner
        shots_df.loc[((shots_df.is_three == True) & (shots_df.angle > -temp_t-9) & (shots_df.angle < -temp_t)), "rbin"] = new_rbin
        shots_df.loc[((shots_df.is_three == True) & (shots_df.angle > -temp_t-9) & (shots_df.angle < -temp_t)), "tbin"] = -temp_t-4.5

        shots_df.loc[((shots_df.is_three == True) & (shots_df.angle > temp_t) & (shots_df.angle < temp_t+9)), "rbin"] = new_rbin
        shots_df.loc[((shots_df.is_three == True) & (shots_df.angle > temp_t) & (shots_df.angle < temp_t+9)), "tbin"] = temp_t+4.5

    return shots_df


def add_polar_visual_assists(fig, three_line_col="orange", restricted_line_col="lightgray"):

    # Add 3pt line
    cnr_angle_min = -np.degrees(np.arctan(5.25 / 22))
    cnr_angle_max = np.degrees(np.arccos(22 / 23.75))
    t = np.linspace(cnr_angle_min, cnr_angle_max, 20)
    r = [22 / np.cos(np.radians(i)) for i in t]

    new_lines = list()
    for i in range(len(t) - 1):
        tmp_dict = dict(type="line", x0=90 - t[i], y0=r[i], x1=90 - t[i + 1], y1=r[i + 1],
                        line=dict(color=three_line_col, width=1), layer='above')
        new_lines.append(tmp_dict)  # Right side of graph / Left on court
        tmp_dict = dict(type="line", x0=-90 + t[i], y0=r[i], x1=-90 + t[i + 1], y1=r[i + 1],
                        line=dict(color=three_line_col, width=1), layer='above')
        new_lines.append(tmp_dict)  # Left side of graph / Right on court
    new_lines.append(
        dict(type="line", x0=-90 + cnr_angle_max, y0=23.75, x1=90 - cnr_angle_max, y1=23.75,
             line=dict(color=three_line_col, width=1), layer='above')
    )

    # Add restricted area line
    new_lines.append(
        dict(type="line", x0=-90, y0=4, x1=90, y1=4,
             line=dict(color=restricted_line_col, width=1), layer='above')
    )
    fig.update_layout(shapes=new_lines)
    return fig


# ==========  Draw similar bits in angular coordinates ==========
yrs = list(range(5, 20))

for yr in yrs[-1:]:

    # ===== Load DataFrame with shot data
    yr_a = ("0" + str(yr))[-2:]
    yr_b = ("0" + str(yr+1))[-2:]
    df_path = f"procdata/shots_df_{yr_a}_{yr_b}.csv"
    shots_df = pd.read_csv(df_path)

    # Generate shot angle & distance data
    shots_df = shots_df.assign(shot_dist_calc=((shots_df.original_x*shots_df.original_x)+((shots_df.original_y)*(shots_df.original_y)))**0.5)
    shots_df = shots_df.assign(angle=(np.arctan2(shots_df.original_x, shots_df.original_y) * 180 / np.pi))
    shots_df.loc[(shots_df["angle"] == -180), "angle"] = 0  # Adjust for weird coordinate system use on point-blank shots -

    # fig = px.histogram(shots_df, x="angle", nbins=100)
    # fig.show(config={'displayModeBar': False})
    #
    # fig = px.scatter(shots_df, x="angle", y="shot_distance")
    # fig.show(config={'displayModeBar': False})

    paper_bgcolor = "wheat"
    plot_bgcolor = "Cornsilk"
    textcolor = "#333333"

    shots_df = add_polar_bins(shots_df)

    # ===== Plot shots_df in the polar coordinate space
    # For the whole NBA
    grp_shots_df = grp_polar_shots(shots_df)
    fig = plot_polar_pps(grp_shots_df)
    fig = viz.add_shotchart_note(fig,
                                 "<B>NBA - shots by angle & distance</B><BR><BR>" +
                                 "'" + yr_a + "/'" + yr_b + " season<BR>" +
                                 "Size: Frequency<BR>Color: Points / 100 shots",
                                 title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
                                 add_sig=False)
    fig = add_polar_visual_assists(fig)
    fig.write_image(f"temp/nba_polar_{yr_a}_{yr_b}.png")
    fig.show(config={'displayModeBar': False})

    # For teams
    teams = ["HOU", "SAS", "OKC", "LAL", "GSW"]
    for team in teams:
        team_df = shots_df[shots_df.team == team]
        grp_shots_df = grp_polar_shots(team_df)
        fig = plot_polar_pps(grp_shots_df)
        fig = viz.add_shotchart_note(fig,
                                     f"<B>{team} - shots by angle & distance</B><BR><BR>" +
                                     "'" + yr_a + "/'" + yr_b + " season<BR>" +
                                     "Size: Frequency<BR>Color: Points / 100 shots",
                                     title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
                                     add_sig=False)
        fig = add_polar_visual_assists(fig)
        fig.write_image(f"temp/{team}_polar_{yr_a}_{yr_b}.png")
        fig.show(config={'displayModeBar': False})

    players = ["Luka Doncic", "James Harden", "LeBron James", "DeMar DeRozan",
               "Chris Paul", "Nikola Jokic", "Damian Lillard"]

    for player in players:
        pl_df = shots_df[shots_df.player == player]
        grp_shots_df = grp_polar_shots(pl_df)
        fig = plot_polar_pps(grp_shots_df)
        fig = viz.add_shotchart_note(fig,
                                     f"<B>{player} - shots by angle & distance</B><BR><BR>" +
                                     "'" + yr_a + "/'" + yr_b + " season<BR>" +
                                     "Size: Frequency<BR>Color: Points / 100 shots",
                                     title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
                                     add_sig=False)
        fig = add_polar_visual_assists(fig)
        fig.write_image(f"temp/{player}_polar_{yr_a}_{yr_b}.png")
        fig.show(config={'displayModeBar': False})

    # fig = px.scatter(grp_shots_df, x="tbin", y="rbin", size="attempts", color="pct",
    #                  color_continuous_scale=px.colors.sequential.Mint, template="plotly_white",
    #                  range_x=[-180, 180], range_y=[-55, 350],
    #                  range_color=[50, 80],
    #                  height=500, width=800, size_max=12
    #                  )
    # fig.update_traces(marker=dict(line=dict(width=0.6, color="#333333"), symbol="circle"))
    # fig.show(config={'displayModeBar': False})

    # fig = px.scatter(grp_shots_df[grp_shots_df["better_side"]==1], x="tbin", y="rbin", size="attempts", color="rel_pct",
    #                  color_continuous_scale=px.colors.diverging.RdYlBu, template="plotly_white",
    #                  range_x=[-180, 180], range_y=[-55, 350],
    #                  range_color=[-1, 1],
    #                  height=500, width=1000, size_max=14,
    #                  )
    # fig.update_traces(marker=dict(line=dict(width=0.6, color="#333333"), symbol="square"))
    # fig.show(config={'displayModeBar': False})

    # fig = px.scatter(grp_shots_df[(grp_shots_df["better_side"]==1) & (grp_shots_df["rel_pct"] > 0.2) & (grp_shots_df["attempts"] > 500)], x="tbin", y="rbin", size="attempts", color="rel_pct",
    #                  color_continuous_scale=px.colors.diverging.RdYlBu, template="plotly_white",
    #                  range_x=[-180, 180], range_y=[-55, 350],
    #                  range_color=[-1, 1],
    #                  height=500, width=1000, size_max=14,
    #                  )
    # fig.update_traces(marker=dict(line=dict(width=0.6, color="#333333"), symbol="square"))
    # fig.show(config={'displayModeBar': False})

    # fig = px.scatter(grp_shots_df[grp_shots_df.better_side != 0], x="tbin", y="rbin", size="attempts", color="rel_pct",
    #                  color_continuous_scale=px.colors.diverging.RdYlBu, template="plotly_white",
    #                  range_x=[-180, 180], range_y=[-5, 35], range_color=[-0.2, 0.2],
    #                  height=900, width=800, size_max=12,
    #                  facet_row="better_side"
    #                  )
    # fig.update_traces(marker=dict(line=dict(width=0.6, color="#333333"), symbol="square"))
    # fig.show(config={'displayModeBar': False})

    # players = ["Luka Doncic", "James Harden", "LeBron James", "DeMar DeRozan",
    #            "Chris Paul", "Nikola Jokic", "Damian Lillard"]
    #
    # for player in players:
    #
    #     # ========================================
    #     # pl_df = shots_df[shots_df["player"] == player]
    #     #
    #     # fig = go.Figure()
    #     # fig.update_layout(
    #     #     margin=dict(l=20, r=20, t=20, b=20),
    #     #     paper_bgcolor=paper_bgcolor,
    #     #     plot_bgcolor=plot_bgcolor,
    #     #     yaxis=dict(
    #     #         scaleanchor="x",
    #     #         scaleratio=1,
    #     #         showgrid=False,
    #     #         zeroline=False,
    #     #         showline=False,
    #     #         ticks='',
    #     #         showticklabels=False,
    #     #         fixedrange=True,
    #     #     )
    #     # )
    #     #
    #     # for make_miss in [0, 1]:
    #     #     if make_miss == 0:
    #     #         symbol = "x"
    #     #         marker_color = "red"
    #     #     else:
    #     #         symbol = "circle"
    #     #         marker_color = "blue"
    #     #
    #     #     xlocs = pl_df[pl_df["shot_made"] == make_miss]["angle"]
    #     #     ylocs = pl_df[pl_df["shot_made"] == make_miss]["shot_distance"]
    #     #
    #     #     fig.add_trace(go.Scatter(
    #     #         x=xlocs, y=ylocs, mode='markers', name='markers',
    #     #         marker=dict(
    #     #             size=8,
    #     #             line=dict(width=0.6, color=textcolor), symbol=symbol,
    #     #             color=marker_color,
    #     #         ),
    #     #         hoverinfo='text'
    #     #     ))
    #     # fig.update_layout(showlegend=False)
    #     # fig = viz.add_shotchart_note(fig,
    #     #                              player +
    #     #                              ": '" + yr_a + "/'" + yr_b + " season<BR><BR>",
    #     #                              title_xloc=0.1, title_yloc=0.885,
    #     #                              size=14, textcolor="#222222")
    #     # fig.show(config={'displayModeBar': False})
    #     # fig.write_image(f"temp/cartesian.png")
    #     # fig = px.scatter(pl_df, x="angle", y="shot_distance", color="shot_made", template="plotly_white")
    #     # fig.show(config={'displayModeBar': False})
    #
    #     # ========================================
    #     # ========================================
    #     pl_df = shots_df[shots_df["player"] == player]
    #     grp_shots_df = grp_polar_shots(pl_df)
    #
    #     tmp_df = grp_shots_df
    #     # tmp_df = grp_shots_df[(grp_shots_df["better_side"] == 1)]
    #     fig = plot_polar_pps(tmp_df)
    #     fig = viz.add_shotchart_note(fig,
    #                                  "<B>" + player + " - shots by angle & distance</B><BR><BR>" +
    #                                  "'" + yr_a + "/'" + yr_b + " season<BR>" +
    #                                  "Size: Frequency<BR>Color: Points / 100 shots",
    #                                  title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
    #                                  add_sig=False)
    #     fig.write_image(f"temp/{player}_polar_{yr_a}_{yr_b}.png")
    #     fig.show(config={'displayModeBar': False})
    #
    #
    #     # fig = px.scatter(grp_shots_df, x="tbin", y="rbin", size="attempts", color="pct",
    #     #                  color_continuous_scale=px.colors.sequential.Mint, template="plotly_white",
    #     #                  range_x=[-180, 180], range_y=[-5, 35],
    #     #                  height=500, width=800, size_max=12
    #     #                  )
    #     # fig.update_traces(marker=dict(line=dict(width=0.6, color="#333333"), symbol="square"))
    #     # fig.show(config={'displayModeBar': False})
    #     # ========================================

    # fig = px.histogram(shots_df, x="original_x", nbins=100)
    # fig.show(config={'displayModeBar': False})
    # fig = px.histogram(shots_df[shots_df["angle"]==-180], x="original_y", nbins=100)
    # fig.show(config={'displayModeBar': False})
