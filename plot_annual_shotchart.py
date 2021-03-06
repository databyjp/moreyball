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

for yr in yrs[-1:]:
    yr_a = ("0" + str(yr))[-2:]
    yr_b = ("0" + str(yr+1))[-2:]
    allshots_df = pd.read_csv(f'procdata/shots_df_{yr_a}_{yr_b}.csv', index_col=0)

    players = ["Luka Doncic", "James Harden", "LeBron James", "DeMar DeRozan",
               "Chris Paul", "Nikola Jokic", "Damian Lillard"]

    for player in players:
    # player = "Luka Doncic"
        shots_df = allshots_df[allshots_df["player"] == player]
        if len(shots_df) > 100:
            fig = viz.plot_hex_shot_chart(shots_df, 'NBA', "All", "pps_abs", title="Shot chart", player=player, mode="light")
            fig = viz.add_shotchart_note(fig,
                                         "<B>Evolution of " + player + "'s<BR>shot distribution</B><BR><BR>" +
                                         "'" + yr_a + "/'" + yr_b + " season<BR><BR>" +
                                         "Size: Shot frequency<BR>Color: Points / 100 shots",
                                         title_xloc=0.1, title_yloc=0.885, size=15, textcolor="#333333")
            fig.show()
            fig.write_image(f"temp/{player}_{yr_a}_{yr_b}.png")

    # fig = viz.plot_hex_shot_chart(allshots_df, 'NBA', "All", "pps_abs", title="Shot chart", mode="light")
    # fig = viz.add_shotchart_note(fig,
    #                              "<B>Evolution of the NBA<BR>by shot distribution</B><BR><BR>" +
    #                              "'" + yr_a + "/'" + yr_b + " season<BR><BR>" +
    #                              "Size: Shot frequency<BR>Color: Points / 100 shots",
    #                              title_xloc=0.1, title_yloc=0.885, size=15, textcolor="#333333")
    #
    # fig.show()
    # fig.write_image(f"temp/frame_{yr_a}_{yr_b}.png")


