# ========== (c) JP Hwang 1/8/20  ==========

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
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

shots_df = pd.read_csv('procdata/shots_df.csv', index_col=0)
all_teams_df = pd.read_csv('procdata/shot_dist_df.csv', index_col=0)

end_yr = dataproc.get_season_yr(shots_df)
seasonyr_str = "'" + str(end_yr-1) + "-'" + str(end_yr)

# ====================================================================
# ========== PLOT LEAGUE WIDE SHOT CHART - ACCURACY ==========
# ====================================================================

team_names = shots_df.team.unique()
shots_teams_list = ['NBA'] + list(np.sort(team_names))
dist_teams_list = ['Leaders'] + list(np.sort(team_names))

# Set parameters
gridsize, min_samples = viz.fill_def_params()
avail_periods = ['All', 1, 2, 3, 4]

latest_data = str(max(pd.to_datetime(shots_df.date)).date())
# last_updated = str(os.path.getmtime('app.py'))

# ====================================================================
# ========== SET UP DASH APP LAYOUT ==========
# ====================================================================
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = html.Div([
    dbc.Jumbotron([
        html.H2("Hindsights on the NBA - Interactive data visualizations & analytics"),
        html.P(
            "Analyse and compare each NBA teams' tendencies, strengths and weaknesses.",
            className="lead",
        ),
        html.Hr(className="my-2"),
        html.P([html.Small(
            "By JP Hwang - Find me on "), html.A(html.Small("twitter"), href="https://twitter.com/_jphwang", title="twitter")
        ])
    ]),
    dbc.Container([
        dbc.Row([
            html.H3('Interactive shot charts'),
        ]),
        dbc.Row([
            dcc.Markdown(
                """
                Use the pulldown to select a team, filter by quarter, and a shot quality measure.

                * **Frequency**: How often a team shoots from a spot, indicated by **size**.

                * **Quality**: Good shot or bad shot? Measured by shot accuracy
                or by points per 100 shots, indicated by **colour**. League avg: ~105.
                """
            ),
        ]),
        dbc.Row(
            html.Div([
                dcc.Dropdown(
                    id='stat-select',
                    options=[
                        {'label': 'Accuracy', 'value': 'acc_abs'},
                        {'label': 'Accuracy vs NBA avg', 'value': 'acc_rel'},
                        {'label': 'Points per shot', 'value': 'pps_abs'}
                    ],
                    value='pps_abs',
                )],
                style={'width': '250px', 'display': 'inline-block'}
            ),
        ),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Dropdown(
                        id='team-select-1',
                        options=[{'label': i, 'value': i} for i in shots_teams_list],
                        value='NBA',
                    )],
                    style={'width': '90px', 'display': 'inline-block'}
                ),
                html.Div([
                    dcc.Dropdown(
                        id='player-select-1',
                        options=[{"label": "All", "value": "All"}],
                        value='All',
                    )],
                    style={'width': '250px', 'display': 'inline-block'}
                ),
                html.Div([
                dcc.Dropdown(
                    id='period-select-1',
                    options=[{'label': i, 'value': i} for i in avail_periods],
                    value=1,
                    )],
                    style={'width': '90px', 'display': 'inline-block'}
                ),
                html.Div([
                    dcc.DatePickerRange(
                        id='date-picker-1',
                        min_date_allowed=min(pd.to_datetime(shots_df.date)).date(),
                        max_date_allowed=max(pd.to_datetime(shots_df.date)).date(),
                        initial_visible_month=max(pd.to_datetime(shots_df.date)).date(),
                        start_date_placeholder_text='From',
                        end_date_placeholder_text='To',
                        clearable=True,
                        # start_date=min(pd.to_datetime(shots_df.date)).date(),
                        # end_date=max(pd.to_datetime(shots_df.date)).date(),
                    )
                ]),
                html.Div([
                    dcc.Dropdown(
                        id='on-court-select-1',
                        options=[{"label": "All", "value": "All"}],
                        value='All',
                    )],
                    style={'width': '250px', 'display': 'inline-block'}
                ),
                html.Div([
                    dcc.Dropdown(
                        id='off-court-select-1',
                        options=[{"label": "All", "value": "All"}],
                        value='All',
                    )],
                    style={'width': '250px', 'display': 'inline-block'}
                ),
                dcc.Graph('shot-chart-1', config={'displayModeBar': False})
            ], lg=6, md=12),
            dbc.Col([
                html.Div([
                    dcc.Dropdown(
                        id='team-select-2',
                        options=[{'label': i, 'value': i} for i in shots_teams_list],
                        value='NBA',
                    )],
                    style={'width': '90px', 'display': 'inline-block'}
                ),
                html.Div([
                    dcc.Dropdown(
                        id='period-select-2',
                        options=[{'label': i, 'value': i} for i in avail_periods],
                        value=4,
                    )],
                    style={'width': '90px', 'display': 'inline-block'}
                ),
                html.Div([
                    dcc.DatePickerRange(
                        id='date-picker-2',
                        min_date_allowed=min(pd.to_datetime(shots_df.date)).date(),
                        max_date_allowed=max(pd.to_datetime(shots_df.date)).date(),
                        initial_visible_month=max(pd.to_datetime(shots_df.date)).date(),
                        start_date_placeholder_text='From',
                        end_date_placeholder_text='To',
                        clearable=True,
                        # start_date=min(pd.to_datetime(shots_df.date)).date(),
                        # end_date=max(pd.to_datetime(shots_df.date)).date(),
                    )
                ]),
                dcc.Graph('shot-chart-2', config={'displayModeBar': False})
            ], lg=6, md=12)]
        ),
    ]),

    dbc.Container([
        html.H3('Related articles'),
        html.Ul([
            html.Li([html.A("Build a web data dashboard in just minutes with Python", href="https://towardsdatascience.com/build-a-web-data-dashboard-in-just-minutes-with-python-d722076aee2b", title="article link")]),
        ])
    ]),
    dbc.Container([
        html.Small('Data from games up to & including: ' + latest_data + '.')
    ]),
    dbc.Container([
        html.P([
            html.Small("© JP Hwang 2020 ("),
            html.A(html.Small("twitter"), href="https://twitter.com/_jphwang", title="twitter"),
            html.Small("), built using Python & Plotly Dash")
        ])
    ]),
])


@app.callback(
    [Output('player-select-1', 'options'), Output('player-select-1', 'value'),
     Output('on-court-select-1', 'options'), Output('on-court-select-1', 'value'),
     Output('off-court-select-1', 'options'), Output('off-court-select-1', 'value')],
    [Input('team-select-1', 'value')]
)
def fill_player_dropdown_1(teamname):

    team_df = shots_df[shots_df["team"] == teamname]
    players = list(team_df["player"].unique())
    pl_list = [{"label": "All", "value": "All"}] + [{"label": i, "value": i} for i in players]

    return pl_list, "All", pl_list, "All", pl_list, "All"


@app.callback(
    Output('shot-chart-1', 'figure'),
    [Input('team-select-1', 'value'), Input('period-select-1', 'value'), Input('stat-select', 'value'),
     Input('date-picker-1', 'start_date'), Input('date-picker-1', 'end_date'),
     Input('player-select-1', 'value'), Input('on-court-select-1', 'value'), Input('off-court-select-1', 'value')]
)
def call_shotchart_1(teamname, period, stat_type, start_date, end_date, player, on_court_a, off_court_a):

    if on_court_a == "All" or on_court_a is None:
        on_court_a = None
    else:
        on_court_a = [on_court_a]

    if off_court_a == "All" or off_court_a is None:
        off_court_a = None
    else:
        off_court_a = [off_court_a]
    fig = viz.plot_hex_shot_chart(shots_df, teamname, period, stat_type, start_date, end_date,
                                  player, on_court_list=on_court_a, off_court_list=off_court_a,
                                  title="Shot chart, " + seasonyr_str)

    return fig


@app.callback(
    Output('shot-chart-2', 'figure'),
    [Input('team-select-2', 'value'), Input('period-select-2', 'value'), Input('stat-select', 'value'),
     Input('date-picker-2', 'start_date'), Input('date-picker-2', 'end_date')]
)
def call_shotchart_2(teamname, period, stat_type, start_date, end_date):

    fig = viz.plot_hex_shot_chart(shots_df, teamname, period, stat_type, start_date, end_date, title="Shot chart, " + seasonyr_str)

    return fig


# @app.callback(
#     Output('shot-dist-graph', 'figure'),
#     [Input('shot-dist-group-select', 'value')]
# )
# def update_shot_dist_chart(grpname):
#
#     fig = viz.make_shot_dist_chart(
#         all_teams_df[all_teams_df.group == grpname], col_col='pl_pps', range_color=[90, 120], size_col='shots_freq')
#     viz.clean_chart_format(fig)
#
#     if len(grpname) > 3:
#         fig.update_layout(height=850, width=1250)
#     else:
#         fig.update_layout(height=500, width=1250)
#
#     return fig


if __name__ == '__main__':
    app.run_server(debug=True)
