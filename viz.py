# ========== (c) JP Hwang 2020-01-10  ==========

import logging
import dataproc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# ===== START LOGGER =====
logger = logging.getLogger(__name__)


def fill_def_params(gridsize=None, min_samples=None):

    if gridsize is None:
        gridsize = 41
    if min_samples is None:
        min_samples = 0.0005

    return gridsize, min_samples


def draw_plotly_court(fig, fig_width=600, margins=10, mode="dark"):

    # mode should be "dark" or "light"

    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

    threept_break_y = 89.47765084

    if mode == "dark":
        three_line_col = "#ffffff"
        main_line_col = "#dddddd"
        paper_bgcolor = "DimGray"
        plot_bgcolor = "black"
    else:
        three_line_col = "#333333"
        main_line_col = "#333333"
        paper_bgcolor = "white"
        plot_bgcolor = "white"

    fig.update_layout(
        # Line Horizontal
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=[
            dict(
                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ),
            dict(
                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ),

            dict(
                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                line=dict(color=main_line_col, width=1),
                fillcolor=main_line_col,
            ),
            dict(
                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
            ),
            dict(
                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color=main_line_col, width=1),
            ),

            dict(type="path",
                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),

            dict(
                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=250, y0=227.5, x1=220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=17.5, x1=80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=27.5, x1=80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=57.5, x1=80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=87.5, x1=80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),

            dict(type="path",
                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),

        ]
    )
    return fig


def get_hexbin_stats(shots_df, gridsize=None, min_samples=None, min_freqs=1):

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from dataproc import get_zones

    matplotlib.use('Agg')

    # TODO - scaling of the hex sizes needs to change for individual players
    # Get parameters
    gridsize, min_samples = fill_def_params(gridsize, min_samples)

    fig, axs = plt.subplots(ncols=2)
    shots_hex = axs[0].hexbin(
        # shots_df.original_x, shots_df.original_y,
        shots_df.halfcourt_x, shots_df.halfcourt_y,
        extent=(-250, 250, 422.5, -47.5), cmap=plt.cm.Reds, gridsize=gridsize)

    makes_df = shots_df[shots_df.shot_made == 1]
    makes_hex = axs[0].hexbin(
        # makes_df.original_x, makes_df.original_y,
        makes_df.halfcourt_x, makes_df.halfcourt_y,
        extent=(-250, 250, 422.5, -47.5), cmap=plt.cm.Reds, gridsize=gridsize)

    assists_df = shots_df[shots_df.assist.notna()]
    assists_hex = axs[0].hexbin(
        # assists_df.original_x, assists_df.original_y,
        assists_df.halfcourt_x, assists_df.halfcourt_y,
        extent=(-250, 250, 422.5, -47.5), cmap=plt.cm.Reds, gridsize=gridsize)
    # plt.close()

    x = [i[0] for i in shots_hex.get_offsets()]
    y = [i[1] for i in shots_hex.get_offsets()]

    zones_list = get_zones(x, y)  # Zones list

    shots_by_hex = shots_hex.get_array()
    shots_by_zones = {i: 0 for i in set(zones_list)}
    makes_by_hex = makes_hex.get_array()
    makes_by_zones = {i: 0 for i in set(zones_list)}
    assists_by_hex = assists_hex.get_array()
    assists_by_zones = {i: 0 for i in set(zones_list)}

    for i in range(len(zones_list)):
        temp_name = zones_list[i]
        shots_by_zones[temp_name] += shots_by_hex[i]
        makes_by_zones[temp_name] += makes_by_hex[i]
        assists_by_zones[temp_name] += assists_by_hex[i]
    accs_by_zones = {k: makes_by_zones[k] / shots_by_zones[k] for k in shots_by_zones.keys()}

    accs_by_hex = np.zeros(len(x))
    # ===== Calculate shot accuracies
    # # raw accuracies
    # accs_by_hex = makes_by_hex/shots_by_hex
    # accs_by_hex[np.isnan(makes_by_hex/shots_by_hex)] = 0
    # # by zones
    # accs_by_hex = np.array([accs_by_zones[zones_list[i]] for i in range(len(zones_list))])
    # by local averaging
    # get closest points for averaging
    xy_df = pd.DataFrame([x, y, makes_by_hex, shots_by_hex]).transpose().rename({0: "x", 1: "y", 2: "makes", 3: "shots"}, axis=1)
    x_spacing = np.sort(xy_df.x.unique())[1] - np.sort(xy_df.x.unique())[0]
    y_spacing = np.sort(xy_df.y.unique())[1] - np.sort(xy_df.y.unique())[0]
    smoothing = 2

    len_df = list()
    for i in range(len(x)):
        tmp_x = x[i]
        tmp_y = y[i]
        filt_xy_df = xy_df[
            ((xy_df.x - tmp_x).abs() < (x_spacing * 2 * smoothing))
            & ((xy_df.y - tmp_y).abs() < (y_spacing * 2 * smoothing))
        ]
        len_df.append(len(filt_xy_df))
        shots_sum = filt_xy_df.shots.sum()
        if shots_sum > 0:
            accs_by_hex[i] = filt_xy_df.makes.sum() / shots_sum
        else:
            accs_by_hex[i] = 0

    ass_perc_by_zones = {k: assists_by_zones[k] / makes_by_zones[k] for k in makes_by_zones.keys()}
    ass_perc_by_hex = np.array([ass_perc_by_zones[zones_list[i]] for i in range(len(zones_list))])

    # accs_by_hex = makes_hex.get_array() / shots_hex.get_array()
    accs_by_hex[np.isnan(accs_by_hex)] = 0  # conver NANs to 0
    ass_perc_by_hex[np.isnan(accs_by_hex)] = 0  # conver NANs to 0

    shot_ev_by_hex = accs_by_hex * 2
    threes_mask = get_threes_mask(gridsize=gridsize)
    for i in range(len(threes_mask)):
        if threes_mask[i]:
            shot_ev_by_hex[i] = shot_ev_by_hex[i] * 1.5

    shots_by_hex = shots_hex.get_array()
    freq_by_hex = shots_by_hex/sum(shots_by_hex)

    x = [i[0] for i in shots_hex.get_offsets()]
    y = [i[1] for i in shots_hex.get_offsets()]

    # ===== FILTER RESULTS BY SAMPLE SIZE =====
    if min_samples is not None:
        if type(min_samples) == float:
            min_samples = max(int(min_samples * len(shots_df)), min_freqs)

        for i in range(len(shots_by_hex)):
            if shots_by_hex[i] < min_samples:
                freq_by_hex[i] = 0

    # ===== END FILTER =====

    hexbin_dict = dict()

    hexbin_dict['xlocs'] = x
    hexbin_dict['ylocs'] = y
    hexbin_dict['shots_by_hex'] = shots_by_hex
    hexbin_dict['freq_by_hex'] = freq_by_hex
    hexbin_dict['accs_by_hex'] = accs_by_hex
    hexbin_dict['ass_perc_by_hex'] = ass_perc_by_hex
    hexbin_dict['shot_ev_by_hex'] = shot_ev_by_hex

    hexbin_dict['gridsize'] = gridsize
    hexbin_dict['n_shots'] = len(shots_df)

    plt.close()

    return hexbin_dict


def clip_hexbin_stats(hexbin_stats, temp_key, minval=None, maxval=None):
    """
    Clip hexbin stats to set hexbin values to max/min based on threshold value
    :param hexbin_stats:
    :param temp_key:
    :param minval:
    :param maxval:
    :return:
    """

    if minval is not None or maxval is not None:
        filter_list = list()
        for i in range(len(hexbin_stats[temp_key])):
            filter_flag = False
            if minval is not None and hexbin_stats[temp_key][i] < minval:
                filter_flag = True
            if maxval is not None and hexbin_stats[temp_key][i] > maxval:
                filter_flag = True
            if filter_flag:
                filter_list.append(i)

        # for k, v in hexbin_stats.items():
        #     if type(v) == np.ndarray:
        #         for i in filter_list:
        #             v[i] = 0
        for i in filter_list:
            hexbin_stats['freq_by_hex'][i] = 0

    else:
        logger.warning('Nothing is going to be clipped! Check your parameters.')

    return hexbin_stats


def filt_hexbins(hexbin_stats, min_threshold=0.0):
    """
    Filter hexbin stats to exclude hexbin values below a threshold value (of frequency)

    :param hexbin_stats:
    :param min_threshold:
    :return:
    """
    from copy import deepcopy

    filt_hexbin_stats = deepcopy(hexbin_stats)
    temp_len = len(filt_hexbin_stats['freq_by_hex'])
    filt_array = [i > min_threshold for i in filt_hexbin_stats['freq_by_hex']]
    for k, v in filt_hexbin_stats.items():
        if type(v) != int:
            # print(k, len(v), temp_len)
            if len(v) == temp_len:
                # print(f'Filtering the {k} array:')
                filt_hexbin_stats[k] = [v[i] for i in range(temp_len) if filt_array[i]]
            else:
                logger.warning(f'WEIRD! The {k} array has a wrong length!')
        else:
            pass
            # print(f'Not filtering {k} as it has an interger value')

    return filt_hexbin_stats


def mark_hexbin_threes(xlocs, ylocs):

    hexbin_isthree = [False] * len(xlocs)
    for i in range(len(xlocs)):
        temp_xloc = xlocs[i]
        temp_yloc = ylocs[i]
        isthree = False
        if temp_xloc < -220 or temp_xloc > 220:
            isthree = True

        shot_dist = (temp_xloc ** 2 + temp_yloc ** 2) ** 0.5
        if shot_dist > 237.5:
            isthree = True

        if isthree:
            hexbin_isthree[i] = True

    return hexbin_isthree


def get_threes_mask(gridsize=51):

    x_coords = list()
    y_coords = list()
    for i in range(-250, 250, 5):
        for j in range(-48, 423, 5):
            x_coords.append(i)
            y_coords.append(j)

    fig, axs = plt.subplots(ncols=2)
    shots_hex = axs[0].hexbin(
        x_coords, y_coords,
        extent=(-250, 250, 422.5, -47.5), cmap=plt.cm.Reds, gridsize=gridsize)
    plt.close()

    xlocs = [i[0] for i in shots_hex.get_offsets()]
    ylocs = [i[1] for i in shots_hex.get_offsets()]

    threes_mask = mark_hexbin_threes(xlocs, ylocs)

    return threes_mask


def plot_shot_hexbins_plotly(
        xlocs, ylocs, freq_by_hex, accs_by_hex,
        marker_cmin=None, marker_cmax=None, colorscale='RdYlBu_r',
        title_txt='', legend_title='Accuracy', fig_width=800,
        hexbin_text=[], ticktexts=[], logo_url=None, show_fig=False, img_out=None,
        mode="dark"):
    """
    Plot shot chart as hexbins
    :param xlocs: list of x locations
    :param ylocs: list of x locations
    :param freq_by_hex: list of shot frequencies, for each location
    :param accs_by_hex: list of shot accuracies, for each location
    :param marker_cmin: min value for marker color range
    :param marker_cmax: max value for marker color range
    :param colorscale: plotly colorscale name
    :param title_txt: plot title text
    :param legend_title: legend title
    :param fig_width: 
    :param hexbin_text:
    :param ticktexts:
    :param logo_url: show team logo?
    :param show_fig:
    :param img_out:
    :return:
    """

    import plotly.graph_objects as go

    if mode == "dark":
        textcolor = "#eeeeee"
    else:
        textcolor = "#222222"

    if marker_cmin is None:
        marker_cmin = min(accs_by_hex)
    if marker_cmax is None:
        marker_cmax = max(accs_by_hex)

    fig = go.Figure()
    draw_plotly_court(fig, fig_width=fig_width)
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

    if show_fig:
        fig.show(
            config={
                'displayModeBar': False
            }
        )
    if img_out is not None:
        fig.write_image(img_out)

    return fig


def add_shotchart_note(fig, title_txt, title_xloc, title_yloc=0.9, size=12):

    textcolor = "#eeeeee"

    fig.update_layout(
        title=dict(
            text=title_txt,
            y=title_yloc,
            x=title_xloc,
            xanchor='left',
            yanchor='middle',
            font=dict(
                # family="Helvetica, Arial, Tahoma",
                size=size,
                color=textcolor
            ),
        ),
        font=dict(
            family="Open Sans, Arial",
            size=14,
            color=textcolor
        ),
        annotations=[
            go.layout.Annotation(
                x=0.5,
                y=0.05,
                showarrow=False,
                text="Twitter: @_jphwang",
                xref="paper",
                yref="paper"
            ),
        ],
    )
    return fig


def plot_parcat_chart(input_df, title_txt='Shot flow - 2018/2019 NBA Season (colored: assisted)', colorscale=[[0, 'gray'], [1, 'crimson']]):

    import pandas as pd
    import plotly.graph_objects as go
    from copy import deepcopy

    # Create new category boolean to color categories
    input_df = input_df.assign(col_cat=1 - 1 * input_df.assist.isna())

    makes_df = input_df[input_df.shot_made == 1]
    temp_assist_col = deepcopy(makes_df.assist.fillna(''))  # Will not sort with NAs in it
    makes_df.loc[:, 'assist'] = temp_assist_col

    i, r = pd.factorize(makes_df['assist'])
    a = np.argsort(np.bincount(i)[i], kind='mergesort')
    makes_df = makes_df.iloc[a]

    i, r = pd.factorize(makes_df['player'])
    a = np.argsort(np.bincount(i)[i], kind='mergesort')
    a = np.flip(a)
    makes_df = makes_df.iloc[a]

    # Create dimensions
    class_dim = go.parcats.Dimension(values=makes_df.player, categoryorder='array', label=" ")
    zone_dim = go.parcats.Dimension(values=makes_df.readable_zone, categoryorder='category ascending', label=" ")
    assist_dim = go.parcats.Dimension(values=makes_df.assist, categoryorder='array', label=" ")

    # Create parcats trace
    color = makes_df.col_cat

    fig = go.Figure(data=[go.Parcats(dimensions=[class_dim, zone_dim, assist_dim],
                                     line={'color': color, 'colorscale': colorscale},
                                     hoveron='color', hoverinfo='count+probability',
                                     labelfont={'size': 13, 'family': "Open Sans, Arial"},
                                     tickfont={'size': 11, 'family': "Open Sans, Arial"}
                                     )])
    # fig.update_layout(
    #     width=900,
    #     height=700,
    #     margin=dict(t=50),
    #     title=dict(
    #         text=title_txt,
    #         y=0.95,
    #         x=0.5,
    #         xanchor='center',
    #         yanchor='middle'),
    #     font=dict(
    #         family="Arial, Tahoma, Helvetica",
    #         size=11,
    #         color="#3f3f3f",
    #     ),
    #     coloraxis=dict(showscale=False),
    #     annotations=[
    #         go.layout.Annotation(
    #             x=0.5,
    #             y=0.0,
    #             showarrow=False,
    #             xanchor='center',
    #             yanchor='middle',
    #             text="Twitter: @_jphwang",
    #             xref="paper",
    #             yref="paper"
    #         ),
    #         go.layout.Annotation(
    #             x=0.5,
    #             y=0.99,
    #             showarrow=False,
    #             xanchor='center',
    #             yanchor='middle',
    #             text="<B>Left:</B> Shooter, <B>Middle:</B> Shot zone, <B>Right:</B> Assist",
    #             xref="paper",
    #             yref="paper"
    #         ),
    #     ],
    # )
    return fig


def make_shot_dist_chart(input_df, color_continuous_scale=None, size_col='shots_count', col_col='pl_acc', range_color=None):

    max_bubble_size = 15
    if color_continuous_scale is None:
        color_continuous_scale = px.colors.diverging.RdYlBu_r
    if range_color is None:
        range_color = [min(input_df[col_col]), max(input_df[col_col])]

    fig = px.scatter(
        input_df, x='min_mid', y='player', size=size_col,
        color=col_col,
        color_continuous_scale=color_continuous_scale,
        range_color=range_color,
        range_x=[0, 49],
        range_y=[-1, len(input_df.player.unique())],
        hover_name='player', hover_data=['min_start', 'min_end', 'shots_count', 'shots_made', 'shots_freq', 'shots_acc', ],
        render_mode='svg'
    )
    fig.update_coloraxes(colorbar=dict(title='Points per<BR>100 shots'))
    fig.update_traces(marker=dict(sizeref=2. * 30 / (max_bubble_size ** 2)))
    fig.update_yaxes(title="Player")
    fig.update_xaxes(title='Minute', tickvals=list(range(0, 54, 6)))

    return fig


def clean_chart_format(fig, add_twitter_name=True):

    annotations = []
    if add_twitter_name:
        annotations.append(
            go.layout.Annotation(
                x=0.9,
                y=1.02,
                showarrow=False,
                text="Twitter: @_jphwang",
                xref="paper",
                yref="paper",
                textangle=0
            ),
        )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations=annotations,
        font=dict(
            family="Open Sans, Arial",
            size=10,
            color="#404040"
        ),
        margin=dict(
            t=20
        )
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='Navy')),
                      selector=dict(mode='markers'))
    fig.update_coloraxes(
        colorbar=dict(
            thicknessmode="pixels", thickness=15,
            outlinewidth=1,
            outlinecolor='#909090',
            lenmode="pixels", len=300,
            yanchor="top",
            y=1,
        ))
    fig.update_yaxes(showgrid=True, gridwidth=1, tickson='boundaries', gridcolor='LightGray', fixedrange=True)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', fixedrange=True)
    return True


def get_rel_stats(rel_hexbin_stats, base_hexbin_stats, min_threshold):

    rel_hexbin_stats['accs_by_hex'] = rel_hexbin_stats['accs_by_hex'] - base_hexbin_stats['accs_by_hex']
    rel_hexbin_stats['shot_ev_by_hex'] = rel_hexbin_stats['shot_ev_by_hex'] - base_hexbin_stats['shot_ev_by_hex']
    rel_hexbin_stats = filt_hexbins(rel_hexbin_stats, min_threshold=min_threshold)

    return rel_hexbin_stats


def plot_hex_shot_chart(shots_df, teamname, period, stat_type,
                        start_date=None, end_date=None, player=None, on_court_list=None, off_court_list=None, gridsize=None, min_samples=None, title=None):

    from copy import deepcopy

    gridsize, min_samples = fill_def_params(gridsize, min_samples)

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

    max_freq = 0.002

    # PLOT OPTIONS: SHOT ACCURACY (ABSOLUTE OR REL VS AVG FROM ZONE); SHOT PPS - ABSOLUTE, OR VS AVG FROM ZONE)

    if stat_type == 'acc_abs':
        hexbin_stats = filt_hexbins(hexbin_stats, min_samples)

        accs_by_hex = hexbin_stats['accs_by_hex']
        colorscale = 'YlOrRd'
        marker_cmin = 0.3
        marker_cmax = 0.6
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
        title_txt=title_txt, hexbin_text=hexbin_text, ticktexts=ticktexts)

    return fig


def plot_raw_shot_chart(shots_df, teamname, period, stat_type,
                        start_date=None, end_date=None, player=None, on_court_list=None, off_court_list=None, title=None, fig_width=600):

    # fig = go.Figure()
    shot_locs_df = shots_df[["converted_x", "converted_y", "shot_made"]]

    fig = px.scatter(shot_locs_df, x="converted_x", y="converted_y", color="shot_made")
    fig = draw_plotly_court(fig, fig_width=fig_width)



    return fig


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)


if __name__ == '__main__':
    main()