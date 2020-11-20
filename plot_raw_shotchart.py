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

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

yrs = list(range(5, 20))

yr = yrs[0]

yr_a = ("0" + str(yr))[-2:]
yr_b = ("0" + str(yr+1))[-2:]
shots_df = pd.read_csv(f'procdata/shots_df_{yr_a}_{yr_b}.csv', index_col=0)




import plotly.express as px
fig = px.scatter(shots_df, x="original_x", y="original_y", color="shot_made", facet_col="halfcourt_conv")
fig.show()


"""
    fig.add_trace(go.Scatter(
        x=xlocs, y=ylocs, mode='markers', name='markers',
        text=hexbin_text,
        marker=dict(
            size=freq_by_hex, sizemode='area', sizeref=2. * max(freq_by_hex) / (18. ** 2), sizemin=1.5,
            color=accs_by_hex, colorscale=colorscale,
            colorbar=dict(
                # thickness=15,
                x=0.88,
                y=0.83,
                thickness=20,
                yanchor='middle',
                len=0.3,
                title=dict(
                    text=legend_title,
                    font=dict(
                        size=14,
                        color=textcolor
                    ),
                ),
                tickvals=[marker_cmin, (marker_cmin + marker_cmax) / 2, marker_cmax],
                ticktext=ticktexts,
                tickfont=dict(
                    size=14,
                    color=textcolor
                )
            ),
            cmin=marker_cmin, cmax=marker_cmax,
            line=dict(width=0.6, color=textcolor), symbol='hexagon',
        ),
        hoverinfo='text'
    ))
"""
