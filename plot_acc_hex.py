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

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

from viz import *
df_path = "procdata/shots_df_19_20.csv"
shots_df = pd.read_csv(df_path)


def new_plot_hex_shot_chart(shots_df, teamname, period, stat_type,
                        start_date=None, end_date=None, player=None, on_court_list=None, off_court_list=None,
                        gridsize=None, min_samples=None, title=None):

    from copy import deepcopy

    gridsize, min_samples = fill_def_params(21, 0.000000)

    league_hexbin_stats = get_hexbin_stats(shots_df, gridsize=gridsize, min_samples=min_samples)

    if player == "All":
        player = None
    temp_df = dataproc.filter_shots_df(shots_df, teamname=teamname, period=period, player=player)

    if on_court_list is not None:
        for on_court in on_court_list:
            temp_df = temp_df[temp_df["on_court"].str.contains(on_court)]

    if off_court_list is not None:
        for off_court in off_court_list:
            temp_df = temp_df[-temp_df["on_court"].str.contains(off_court)]

    if start_date is not None:
        temp_df = temp_df[pd.to_datetime(temp_df.date) >= start_date]

    if end_date is not None:
        temp_df = temp_df[pd.to_datetime(temp_df.date) <= end_date]

    hexbin_stats = get_hexbin_stats(temp_df, gridsize=gridsize, min_samples=min_samples)
    title_suffix = teamname + ', Qtr: ' + str(period)

    max_freq = 0.000004

    # PLOT OPTIONS: SHOT ACCURACY (ABSOLUTE OR REL VS AVG FROM ZONE); SHOT PPS - ABSOLUTE, OR VS AVG FROM ZONE)

    if stat_type == 'acc_abs':
        hexbin_stats = filt_hexbins(hexbin_stats, min_samples)

        accs_by_hex = hexbin_stats['accs_by_hex']
        colorscale = 'YlOrRd'
        marker_cmin = 0.4
        marker_cmax = 0.5
        legend_title = 'Accuracy'
        title_suffix += '<BR>Shot accuracy'

        freq_by_hex = np.array([min(max_freq, i) for i in hexbin_stats['freq_by_hex']])
        hexbin_text = [
            '<i>Accuracy: </i>' + str(round(accs_by_hex[i] * 100, 1)) + '%<BR>'
            for i in range(len(freq_by_hex))
        ]
        ticktexts = [str(marker_cmin * 100) + '%-', "", str(marker_cmax * 100) + '%+']

    elif stat_type == 'acc_rel':
        rel_hexbin_stats = deepcopy(hexbin_stats)
        base_hexbin_stats = deepcopy(league_hexbin_stats)
        hexbin_stats = get_rel_stats(rel_hexbin_stats, base_hexbin_stats, min_samples)

        accs_by_hex = hexbin_stats['accs_by_hex']
        colorscale = 'RdYlBu_r'
        marker_cmin = -0.1
        marker_cmax = 0.1
        legend_title = 'Accuracy'
        title_suffix += '<BR>Shot accuracy vs NBA average'

        freq_by_hex = np.array([min(max_freq, i) for i in hexbin_stats['freq_by_hex']])
        hexbin_text = [
            '<i>Accuracy: </i>' + str(round(accs_by_hex[i] * 100, 1)) + '%<BR>'
            for i in range(len(freq_by_hex))
        ]
        ticktexts = [str(marker_cmin * 100) + '%-', "", str(marker_cmax * 100) + '%+']

    elif stat_type == 'pps_abs':
        hexbin_stats = filt_hexbins(hexbin_stats, min_samples)

        accs_by_hex = hexbin_stats['shot_ev_by_hex']
        colorscale = 'RdYlBu_r'
        # colorscale = 'Portland'
        marker_cmin = 0.75  # 1.048 -> leage avg
        marker_cmax = 1.25
        legend_title = 'PTS/100'
        title_suffix += '<BR>Expected points per 100 shots'

        freq_by_hex = np.array([min(max_freq, i) for i in hexbin_stats['freq_by_hex']])
        hexbin_text = [
            '<i>Point per 100 shots: </i>' + str(round(accs_by_hex[i] * 100, 1))
            for i in range(len(freq_by_hex))
        ]
        ticktexts = [str(int(marker_cmin * 100)) + '-', "", str(int(marker_cmax * 100)) + '+']

    # elif stat_type == 'pps_rel':
    #     accs_by_hex = hexbin_stats['shot_ev_by_hex']
    #     colorscale = 'RdYlBu_r'
    #     marker_cmin = 0.848
    #     marker_cmax = 1.248

    xlocs = hexbin_stats['xlocs']
    ylocs = hexbin_stats['ylocs']

    if title is not None:
        title_txt = title
        if len(title_suffix) is not None:
            title_txt += '<BR>' + title_suffix

    fig = plot_shot_hexbins_plotly(
        xlocs, ylocs, freq_by_hex, accs_by_hex,
        marker_cmin, marker_cmax, colorscale=colorscale, legend_title=legend_title,
        title_txt=title_txt, hexbin_text=hexbin_text, ticktexts=ticktexts, size_factor=37)

    return fig


fig = new_plot_hex_shot_chart(shots_df, 'NBA', "All", "acc_abs", title="Shot chart")
fig.show()
fig.write_image(f"temp/league_avg.png")


# TODO - what does the shot accuracy chart look like as a funciton of angle & distance?
# What does it look like as a league?
# Lefties vs righties?

