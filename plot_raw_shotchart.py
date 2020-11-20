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
import plotly.graph_objects as go
import viz

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

yrs = list(range(5, 20))

yr = yrs[-1]

yr_a = ("0" + str(yr))[-2:]
yr_b = ("0" + str(yr+1))[-2:]
allshots_df = pd.read_csv(f'procdata/shots_df_{yr_a}_{yr_b}.csv', index_col=0)

fig_width = 800
textcolor = "#ffffff"
textcolor = "#333333"

# # ========== PLOT ALL MADE SHOTS / ALL MISSED SHOTS SEPARATELY ==========
# shots_df = allshots_df
#
# for make_miss in [0, 1]:
#
#     fig = go.Figure()
#     viz.draw_plotly_court(fig, fig_width=fig_width, mode="light")
#     if make_miss == 0:
#         symbol = "x"
#         marker_color = "red"
#         annotation = "NBA" + ": '" + yr_a + "/'" + yr_b + " season<BR><BR>Missed shots"
#     else:
#         symbol = "circle"
#         marker_color = "blue"
#         annotation = "NBA" + ": '" + yr_a + "/'" + yr_b + " season<BR><BR>Made shots"
#
#     xlocs = shots_df[shots_df["shot_made"] == make_miss]["original_x"]
#     ylocs = shots_df[shots_df["shot_made"] == make_miss]["original_y"]
#
#     fig.add_trace(go.Scatter(
#         x=xlocs, y=ylocs, mode='markers', name='markers',
#         marker=dict(
#             size=2,
#             line=dict(width=0.6, color=textcolor), symbol=symbol,
#             color=marker_color,
#         ),
#         hoverinfo='text'
#     ))
#     fig.update_layout(showlegend=False)
#     fig = viz.add_shotchart_note(fig,
#                                  annotation,
#                                  title_xloc=0.1, title_yloc=0.885,
#                                  size=14, textcolor="#222222")
#     fig.show(config={'displayModeBar': False})
#     fig.write_image(f"temp/NBA_{yr_a}_{yr_b}_raw_shot_{make_miss}.png")
#
# # ==================================================

# ========== PLOT ALL NBA SHOTS ==========
shots_df = allshots_df
fig = go.Figure()
viz.draw_plotly_court(fig, fig_width=fig_width, mode="light")

for make_miss in [0, 1]:

    if make_miss == 0:
        symbol = "x"
        marker_color = "red"
    else:
        symbol = "circle"
        marker_color = "blue"

    xlocs = shots_df[shots_df["shot_made"] == make_miss]["original_x"]
    ylocs = shots_df[shots_df["shot_made"] == make_miss]["original_y"]

    fig.add_trace(go.Scatter(
        x=xlocs, y=ylocs, mode='markers', name='markers',
        marker=dict(
            size=2,
            line=dict(width=0.6, color=textcolor), symbol=symbol,
            color=marker_color,
        ),
        hoverinfo='text'
    ))
fig.update_layout(showlegend=False)
annotation = "NBA" + ": '" + yr_a + "/'" + yr_b + " season<BR><BR>All shots"
fig = viz.add_shotchart_note(fig,
                             annotation,
                             title_xloc=0.1, title_yloc=0.885,
                             size=14, textcolor="#222222")
fig.show(config={'displayModeBar': False})
fig.write_image(f"temp/NBA_{yr_a}_{yr_b}_raw_shot.png")

# ==================================================

# ========== PLOT PLAYERS' SHOTS ==========
players = ["LeBron James", "Luka Doncic", "Kevin Durant", "Joel Embiid", "James Harden",
           "DeMar DeRozan", "Ben Simmons", "Chris Paul", "Nikola Jokic", "Rudy Gobert", "Damian Lillard"]

for player in players:
    shots_df = allshots_df[allshots_df["player"] == player]

    fig = go.Figure()
    viz.draw_plotly_court(fig, fig_width=fig_width, mode="light")

    for make_miss in [0, 1]:

        if make_miss == 0:
            symbol = "x"
            marker_color = "red"
        else:
            symbol = "circle"
            marker_color = "blue"

        xlocs = shots_df[shots_df["shot_made"] == make_miss]["original_x"]
        ylocs = shots_df[shots_df["shot_made"] == make_miss]["original_y"]

        fig.add_trace(go.Scatter(
            x=xlocs, y=ylocs, mode='markers', name='markers',
            marker=dict(
                size=8,
                line=dict(width=0.6, color=textcolor), symbol=symbol,
                color=marker_color,
            ),
            hoverinfo='text'
        ))
    fig.update_layout(showlegend=False)
    fig = viz.add_shotchart_note(fig,
                                 player +
                                 ": '" + yr_a + "/'" + yr_b + " season<BR><BR>",
                                 title_xloc=0.1, title_yloc=0.885,
                                 size=14, textcolor="#222222")
    fig.show(config={'displayModeBar': False})
    fig.write_image(f"temp/{player}_{yr_a}_{yr_b}_raw.png")
