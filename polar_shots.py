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
import dataproc
import viz

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

fig_width = 600


# ==========  Draw similar bits in angular coordinates ==========
yrs = list(range(5, 20))

for yr in yrs[:]:

    # ===== Load DataFrame with shot data
    yr_a = ("0" + str(yr))[-2:]
    yr_b = ("0" + str(yr+1))[-2:]
    df_path = f"procdata/shots_df_{yr_a}_{yr_b}.csv"
    shots_df = pd.read_csv(df_path)
    shots_df = shots_df[shots_df.data_set.str.contains("Regular")]

    shots_df = dataproc.add_polar_columns(shots_df)

    # fig = px.histogram(shots_df, x="angle", nbins=100)
    # fig.show(config={'displayModeBar': False})
    #
    # fig = px.scatter(shots_df, x="angle", y="shot_distance")
    # fig.show(config={'displayModeBar': False})

    paper_bgcolor = "wheat"
    plot_bgcolor = "Cornsilk"
    textcolor = "#333333"

    shots_df = dataproc.add_polar_bins(shots_df, large_tbins=False)

    # ===== Plot shots_df in the polar coordinate space
    # For the whole NBA
    grp_shots_df = dataproc.grp_polar_shots(shots_df)
    fig = viz.plot_polar_pps(grp_shots_df)
    fig = viz.add_shotchart_note(fig,
                                 "<B>NBA - shots by angle & distance</B><BR><BR>" +
                                 "'" + yr_a + "/'" + yr_b + " season<BR>" +
                                 "Size: Frequency<BR>Color: Points / 100 shots",
                                 title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
                                 add_sig=False)
    fig = viz.add_polar_visual_assists(fig)
    fig.show(config={'displayModeBar': False})
    fig.write_image(f"temp/nba_polar_{yr_a}_{yr_b}.png")

    # For teams
    teams = ["HOU", ""]
    for team in teams:
        team_df = shots_df[shots_df.team == team]
        grp_shots_df = dataproc.grp_polar_shots(team_df)
        fig = viz.plot_polar_pps(grp_shots_df)
        fig = viz.add_shotchart_note(fig,
                                     f"<B>{team} - shots by angle & distance</B><BR><BR>" +
                                     "'" + yr_a + "/'" + yr_b + " season<BR>" +
                                     "Size: Frequency<BR>Color: Points / 100 shots",
                                     title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
                                     add_sig=False)
        fig = viz.add_polar_visual_assists(fig)
        fig.show(config={'displayModeBar': False})
        fig.write_image(f"temp/{team}_polar_{yr_a}_{yr_b}.png")

        crunch_df = shots_df[(np.abs(shots_df.away_score - shots_df.home_score) <= 8) & (shots_df.period >= 4) & (shots_df.team == team)]
        grp_shots_df = dataproc.grp_polar_shots(crunch_df)
        fig = viz.plot_polar_pps(grp_shots_df)
        fig = viz.add_shotchart_note(fig,
                                     f"<B>{team} - crunch time (score within 8 in 4Q+) shots by angle & distance</B><BR><BR>" +
                                     "'" + yr_a + "/'" + yr_b + " season<BR>" +
                                     "Size: Frequency<BR>Color: Points / 100 shots",
                                     title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
                                     add_sig=False)
        fig = viz.add_polar_visual_assists(fig)
        fig.show(config={'displayModeBar': False})
        fig.write_image(f"temp/{team}_crunchtime_polar_{yr_a}_{yr_b}.png")

    # # Show on/off offence graphs for players
    # for (team, player) in [("HOU", "Harden"), ("PHI", "Joel Embiid"), ("LAL", "LeBron James"), ("MIL", "Giannis")]:
    #     team_df = shots_df[shots_df.team == team]
    #
    #     for on_state in [1, -1]:
    #         if on_state == 1:
    #             filt_df = team_df[team_df.on_court.notna()]
    #             filt_df = filt_df[filt_df.on_court.str.contains(player)]
    #             on_txt = "With "
    #         else:
    #             filt_df = team_df[team_df.on_court.notna()]
    #             filt_df = filt_df[-filt_df.on_court.str.contains(player)]
    #             on_txt = "Without "
    #
    #         grp_shots_df = dataproc.grp_polar_shots(filt_df)
    #         fig = viz.plot_polar_pps(grp_shots_df)
    #         fig = viz.add_shotchart_note(fig,
    #                                      f"<B>{team} - Shots by angle & distance</B><BR><BR>" +
    #                                      f"{on_txt}{player} - {len(filt_df)} shots<BR>" +
    #                                      "'" + yr_a + "/'" + yr_b + " season<BR>" +
    #                                      "Size: Frequency<BR>Color: Points / 100 shots",
    #                                      title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
    #                                      add_sig=False)
    #         fig = viz.add_polar_visual_assists(fig)
    #         fig.show(config={'displayModeBar': False})
    #
    # # Show on/off defence graphs for players
    # tm_pls = [("UTA", "Gobert"), ("PHI", "Joel Embiid"), ("LAL", "Anthony Davis"), ("GSW", "Draymond Green"), ("CLE", "Kevin Love")]
    # for (team, player) in tm_pls:
    #     team_games = shots_df[shots_df.team == team].game_id.unique()
    #     team_games_df = shots_df[shots_df.game_id.isin(team_games)]
    #     opp_df = team_games_df[team_games_df.team != team]
    #
    #     for on_state in [1, -1]:
    #         if on_state == 1:
    #             filt_df = opp_df[opp_df.on_court.notna()]
    #             filt_df = filt_df[filt_df.on_court.str.contains(player)]
    #             on_txt = "With "
    #         else:
    #             filt_df = opp_df[opp_df.on_court.notna()]
    #             filt_df = filt_df[-filt_df.on_court.str.contains(player)]
    #             on_txt = "Without "
    #
    #         grp_shots_df = dataproc.grp_polar_shots(filt_df)
    #         fig = viz.plot_polar_pps(grp_shots_df)
    #         fig = viz.add_shotchart_note(fig,
    #                                      f"<B>{team} - Opponent shots by angle & distance</B><BR><BR>" +
    #                                      f"{on_txt}{player} - {len(filt_df)} shots<BR>" +
    #                                      "'" + yr_a + "/'" + yr_b + " season<BR>" +
    #                                      "Size: Frequency<BR>Color: Points / 100 shots",
    #                                      title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
    #                                      add_sig=False)
    #         fig = viz.add_polar_visual_assists(fig)
    #         fig.show(config={'displayModeBar': False})
    #
    # # For players
    # players = ["Rudy Gobert", "Luka Doncic", "James Harden", "LeBron James", "DeMar DeRozan",
    #            "Chris Paul", "Trae Young", "Damian Lillard"]
    #
    # for player in players:
    #     pl_df = shots_df[shots_df.player == player]
    #     grp_shots_df = dataproc.grp_polar_shots(pl_df)
    #     fig = viz.plot_polar_pps(grp_shots_df)
    #     fig = viz.add_shotchart_note(fig,
    #                                  f"<B>{player} - shots by angle & distance</B><BR><BR>" +
    #                                  "'" + yr_a + "/'" + yr_b + " season<BR>" +
    #                                  "Size: Frequency<BR>Color: Points / 100 shots",
    #                                  title_xloc=0.085, title_yloc=0.915, size=13, textcolor="#333333",
    #                                  add_sig=False)
    #     fig = viz.add_polar_visual_assists(fig)
    #     fig.show(config={'displayModeBar': False})
    #     fig.write_image(f"temp/{player}_polar_{yr_a}_{yr_b}.png")

