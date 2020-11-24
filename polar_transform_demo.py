# ========== (c) JP Hwang 23/11/20  ==========

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
import plotly.graph_objects as go

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

# ==========  Draw court in cartesian coordiantes ==========
fig_width = 700

fig = go.Figure()
fig = viz.draw_plotly_court(fig, fig_width=fig_width, mode="light")
fig.show(config={'displayModeBar': False})

# Highlight areas around the rim & past the 3
fig = go.Figure()
fig = viz.draw_plotly_court(fig, fig_width=fig_width, mode="light")
fig.add_shape(
    dict(type="path",
         path=viz.ellipse_arc(a=40, b=40, start_angle=-np.pi, end_angle=np.pi),
         line=dict(color="gray", width=1), layer='above'),
)
fig.add_shape(
    dict(type="path",
         path=viz.ellipse_arc(a=277.5, b=277.5, start_angle=-np.pi, end_angle=np.pi),
         line=dict(color="gray", width=1), layer='above'),
)
# fig.show(config={'displayModeBar': False})

# Draw marker shapes
fig = go.Figure()
fig = viz.draw_plotly_court(fig, fig_width=fig_width, mode="light")
angle_offset = 22.5
fig.add_shape(
    dict(type="line", x0=0, y0=0, x1=np.cos(np.pi*7/16)*277.5, y1=np.sin(np.pi*7/16)*277.5,
         line=dict(color="gray", width=1),
         layer='above'),
)
fig.add_shape(
    dict(type="line", x0=0, y0=0, x1=np.cos(np.pi*9/16)*277.5, y1=np.sin(np.pi*9/16)*277.5,
         line=dict(color="gray", width=1),
         layer='above'),
)
fig.add_shape(
    dict(type="path",
         path=viz.ellipse_arc(a=277.5, b=277.5, start_angle=np.pi*7/16, end_angle=np.pi*9/16),
         line=dict(color="gray", width=1), layer='above'),
)
fig.add_shape(
    dict(type="path",
         path=viz.ellipse_arc(a=257.5, b=257.5, start_angle=-np.pi, end_angle=np.pi),
         line=dict(color="gray", width=1), layer='above'),
)
fig.add_shape(
    dict(type="path",
         path=viz.ellipse_arc(a=237.5, b=237.5, start_angle=-np.pi, end_angle=np.pi),
         line=dict(color="gray", width=1), layer='above'),
)
fig.show(config={'displayModeBar': False})
# ========================================

# ==========  Draw similar bits in angular coordinates ==========
fig = go.Figure()

three_line_col = "blue"
main_line_col = "#333333"
paper_bgcolor = "wheat"
plot_bgcolor = "Cornsilk"
fig.update_layout(
    # Line Horizontal
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
        showticklabels=False,
        fixedrange=True,
        zerolinewidth=0.2,
        zerolinecolor="#dddddd"
    ),
    xaxis=dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        ticks='',
        showticklabels=False,
        fixedrange=True,
        zerolinewidth=0.2,
        zerolinecolor="#dddddd"
    ),
    shapes=[
        dict(
            type="rect", x0=-22.5/2, y0=0, x1=22.5/2, y1=4,
            line=dict(color=main_line_col, width=0.5),
            layer='above'
        ),
        dict(
            type="rect", x0=-22.5/2, y0=0, x1=22.5/2, y1=23.75,
            line=dict(color=main_line_col, width=0.5),
            layer='above'
        ),
        dict(
            type="rect", x0=-22.5/2, y0=0, x1=22.5/2, y1=27.75,
            line=dict(color=main_line_col, width=0.5),
            layer='above'
        ),
    ]
)


def format_chart(fig, xrange=[-120, 120], yrange=[-4, 40], margins=10):

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_xaxes(range=[xrange[0] - margins, xrange[1] + margins])
    fig.update_yaxes(range=[yrange[0] - margins, yrange[1] + margins])
    fig.update_layout(width=fig_width, height=fig_height)
    return fig


fig = format_chart(fig)
# fig.show(config={'displayModeBar': False})
# ========================================

# ==================== Show transformed court w/ lines ====================
fig = go.Figure()

three_line_col = "blue"
main_line_col = "#333333"
light_line_col = "LightGray"
paper_bgcolor = "wheat"
plot_bgcolor = "Cornsilk"
fig.update_layout(
    # Line Horizontal
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
        # showticklabels=False,
        fixedrange=True,
        zerolinewidth=0.2,
        zerolinecolor="#dddddd"
    ),
    xaxis=dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        ticks='',
        # showticklabels=False,
        fixedrange=True,
        zerolinewidth=0.2,
        zerolinecolor="#dddddd"
    ),
)
# 3 pt line
# Corner 3pt line: r=22 * cos(theta), -13.42 deg <= theta <= 22.13deg (from horizontal)
cnr_angle_min = -np.degrees(np.arctan(5.25/22))
cnr_angle_max = np.degrees(np.arccos(22/23.75))
t = np.linspace(cnr_angle_min, cnr_angle_max, 20)
r = [22 / np.cos(np.radians(i)) for i in t]

new_lines = list()
for i in range(len(t)-1):
    tmp_dict = dict(type="line", x0=90-t[i], y0=r[i], x1=90-t[i+1], y1=r[i+1],
                    line=dict(color=three_line_col, width=1), layer='above')
    new_lines.append(tmp_dict)  # Right side of graph / Left on court
    tmp_dict = dict(type="line", x0=-90+t[i], y0=r[i], x1=-90+t[i+1], y1=r[i+1],
                    line=dict(color=three_line_col, width=1), layer='above')
    new_lines.append(tmp_dict)  # Left side of graph / Right on court
new_lines.append(
    dict(type="line", x0=-90+cnr_angle_max, y0=23.75, x1=90-cnr_angle_max, y1=23.75,
         line=dict(color=three_line_col, width=1), layer='above')
)

# Court boundary - sides
cnr_angle_min = -np.degrees(np.arctan(5.25/25))
cnr_angle_max = np.degrees(np.arctan(41.75/25))
t = np.linspace(cnr_angle_min, cnr_angle_max, 20)
r = [25 / np.cos(np.radians(i)) for i in t]

for i in range(len(t)-1):
    tmp_dict = dict(type="line", x0=90-t[i], y0=r[i], x1=90-t[i+1], y1=r[i+1],
                    line=dict(color=light_line_col, width=1), layer='above')
    new_lines.append(tmp_dict)  # Right side of graph / Left on court
    tmp_dict = dict(type="line", x0=-90+t[i], y0=r[i], x1=-90+t[i+1], y1=r[i+1],
                    line=dict(color=light_line_col, width=1), layer='above')
    new_lines.append(tmp_dict)  # Left side of graph / Right on court

# Court boundary - top line
cnr_angle_min = np.degrees(np.arctan(41.75/25))
cnr_angle_max = 90
t = np.linspace(cnr_angle_min, cnr_angle_max, 20)
r = [41.75 / np.sin(np.radians(i)) for i in t]

for i in range(len(t)-1):
    tmp_dict = dict(type="line", x0=90-t[i], y0=r[i], x1=90-t[i+1], y1=r[i+1],
                    line=dict(color=light_line_col, width=1), layer='above')
    new_lines.append(tmp_dict)  # Right side of graph / Left on court
    tmp_dict = dict(type="line", x0=-90+t[i], y0=r[i], x1=-90+t[i+1], y1=r[i+1],
                    line=dict(color=light_line_col, width=1), layer='above')
    new_lines.append(tmp_dict)  # Left side of graph / Right on court

# Court boundary - bottom line
cnr_angle_min = np.degrees(np.arctan(-5.25/25))
cnr_angle_max = -90
t = np.linspace(cnr_angle_min, cnr_angle_max, 20)
r = [abs(5.25 / np.sin(np.radians(i))) for i in t]

for i in range(len(t)-1):
    tmp_dict = dict(type="line", x0=90-t[i], y0=r[i], x1=90-t[i+1], y1=r[i+1],
                    line=dict(color=light_line_col, width=1), layer='above')
    new_lines.append(tmp_dict)  # Right side of graph / Left on court
    tmp_dict = dict(type="line", x0=-90+t[i], y0=r[i], x1=-90+t[i+1], y1=r[i+1],
                    line=dict(color=light_line_col, width=1), layer='above')
    new_lines.append(tmp_dict)  # Left side of graph / Right on court

# Add restricted area semicircle
new_lines.append(
    dict(type="line", x0=-90, y0=4, x1=90, y1=4,
         line=dict(color="orange", width=1), layer='above')
)

fig.update_layout(shapes=new_lines)

fig = format_chart(fig, xrange=[-180, 180], yrange=[-2, 50], margins=0)
fig.show(config={'displayModeBar': False})
