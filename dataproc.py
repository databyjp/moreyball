# ========== (c) JP Hwang 1/8/20  ==========

import logging
import pandas as pd

# ===== START LOGGER =====
logger = logging.getLogger(__name__)


def load_log_df(logpath):

    log_df = pd.read_csv(logpath)

    return log_df


def load_df_from_logfile_list(logdir, logfiles_list, testmode=False):

    import os

    temp_list = list()
    if testmode:
        logfiles_list = logfiles_list[:100]

    for f in logfiles_list:
        try:
            logger.info(f'Loading & processing {f}')
            logpath = os.path.join(logdir, f)
            log_df = load_log_df(logpath)
            log_df['logfile'] = f
            temp_list.append(log_df)
        except:
            logger.error(f'Weird, error loading {f}')

    log_df = pd.concat(temp_list, axis=0, ignore_index=True)

    return log_df


def mark_df_threes(in_df):

    # Mark threes
    # 22 ft - corner 3s
    in_df.loc[:, 'is_three'] = False

    if 'simple_zone' not in in_df.columns and 'shot_zone' not in in_df.columns:
        in_df.loc[
            (
                    (in_df.original_x < -220) &
                    ((in_df.event_type == 'shot') | (in_df.event_type == 'miss'))
            )
            , 'is_three'] = True
        in_df.loc[
            (
                    (in_df.original_x > 220) &
                    ((in_df.event_type == 'shot') | (in_df.event_type == 'miss'))
            )
            , 'is_three'] = True

        # 23.75 ft - 3 pt arc
        in_df.loc[
            (
                    (in_df.shot_distance >= 23.75) &
                    ((in_df.event_type == 'shot') | (in_df.event_type == 'miss'))
            )
            , 'is_three'] = True
    else:
        for zone_col in ['simple_zone', 'shot_zone']:
            if zone_col in in_df.columns:
                for zone_txt in ['Short 3', '3s', '30+']:
                    in_df.loc[
                        (in_df[zone_col].str.contains(zone_txt))
                        , 'is_three'] = True

    return in_df


def flip_x_coords(log_df):

    log_df = log_df.assign(unflipped_x=log_df.original_x)
    log_df = log_df.assign(original_x=-log_df.original_x)

    return log_df


def filter_error_rows(log_df, filt_cols=('original_x', 'original_y', 'shot_distance')):

    for filt_col in filt_cols:
        log_df = log_df[log_df[filt_col].apply(lambda x: not isinstance(x, str))]
        log_df = log_df[log_df[filt_col].notna()]

    return log_df


def filter_oncourt_pl(in_df, playername, playeron=True, exclude_player=True):

    oncourt_names = (
            in_df['a1'] + in_df['a2'] + in_df['a3'] + in_df['a4'] + in_df['a5']
            + in_df['h1'] + in_df['h2'] + in_df['h3'] + in_df['h4'] + in_df['h5']
    )
    # in_df['oncourt'] = oncourt_names
    in_df = in_df.assign(oncourt=oncourt_names)

    player_filter = in_df.oncourt.str.contains(playername)
    if playeron is False:
        player_filter = -player_filter

    in_df = in_df[player_filter]

    in_df = in_df.drop(labels='oncourt', axis=1)

    if exclude_player:
        in_df = in_df[in_df['player'] != playername]

    return in_df


def get_zones(x, y, excl_angle=False):

    def append_name_by_angle(temp_angle):

        if excl_angle:
            temp_text = ''
        else:
            if temp_angle < 60 and temp_angle >= -90:
                temp_text = '_right'
            elif temp_angle < 120 and temp_angle >= 60:
                temp_text = '_middle'
            else:
                temp_text = '_left'
        return temp_text

    import math

    zones_list = list()
    for i in range(len(x)):

        temp_angle = math.atan2(y[i], x[i]) / math.pi * 180
        temp_dist = ((x[i] ** 2 + y[i] ** 2) ** 0.5) / 10

        if temp_dist > 30:
            zone = '7 - 30+ ft'
        elif (x[i] < -220 or x[i] > 220) and y[i] < 90:
            zone = '4 - Corner 3s'
            zone += append_name_by_angle(temp_angle)
        elif temp_dist > 27:
            zone = '6 - Long 3s'
            zone += append_name_by_angle(temp_angle)
        elif temp_dist > 23.75:
            zone = '5 - Short 3 (<27 ft)'
            zone += append_name_by_angle(temp_angle)
        elif temp_dist > 14:
            zone = '3 - Long 2 (14+ ft)'
            zone += append_name_by_angle(temp_angle)
        elif temp_dist > 4:
            zone = '2 - Short 2 (4-14 ft)'
            zone += append_name_by_angle(temp_angle)
        else:
            zone = '1 - Within 4 ft'

        zones_list.append(zone)

    return zones_list


def get_season_yr(log_df):

    import re
    yr_nums = re.findall("[0-9]+", log_df.iloc[0]["data_set"]) + re.findall("[0-9]+", log_df.iloc[-1]["data_set"])
    yr_nums = [yr[:-2] for yr in yr_nums]
    max_yr = int(max(yr_nums))

    return max_yr


def process_shots_df(log_df):

    import pandas as pd
    """
    :param log_df:
    :return:
    """

    # Filter out rows where shots are string values or unknown
    log_df = filter_error_rows(log_df)

    # Add 'shots_made' column - boolean for shot being made
    log_df['shot_made'] = 0
    log_df['shot_made'] = log_df.shot_made.mask(log_df.event_type == 'shot', 1)

    shots_df = log_df[(log_df.event_type == 'shot') | (log_df.event_type == 'miss')]

    if 'unflipped_x' not in shots_df.columns:
        logger.info('Flipping x_coordinates because they are reversed somehow :(')
        shots_df = flip_x_coords(shots_df)

    shots_df = shots_df.assign(shot_zone=get_zones(list(shots_df['original_x']), list(shots_df['original_y'])))
    shots_df = shots_df.assign(simple_zone=get_zones(list(shots_df['original_x']), list(shots_df['original_y']), excl_angle=True))
    shots_df = mark_df_threes(shots_df)

    # Set up total time (game time) column & score difference column
    remaining_time = [i.split(':') for i in list(shots_df['remaining_time'])]
    tot_time = list()
    for i in range(len(shots_df)):
        if shots_df.iloc[i]['period'] < 5:
            tmp_gametime = ((shots_df.iloc[i]['period'] - 1) * 12) + (11 - int(remaining_time[i][1])) + (1 - (int(remaining_time[i][2]) / 60))
        else:
            tmp_gametime = 48.0 + ((shots_df.iloc[i]['period'] - 5) * 5) + (4 - int(remaining_time[i][1])) + (1 - (int(remaining_time[i][2]) / 60))
        tot_time.append(tmp_gametime)
    shots_df['tot_time'] = tot_time
    shots_df = shots_df.assign(score_diff=abs(shots_df.home_score - shots_df.away_score))

    # Mark garbage time shots: Rule: up 13 with a minute left, increasing by one each minute
    garbage_marker = pd.Series([False] * len(shots_df))
    for i, row in shots_df.iterrows():
        if (row["period"] == 4):
            rem_mins = row["remaining_time"].split(":")[1]
            score_threshold = 13 + int(rem_mins)
            if row["score_diff"] >= score_threshold:
                garbage_marker[i] = True
    shots_df['garbage'] = garbage_marker

    return shots_df


def load_latest_logfile(logfile_dir):

    import re
    import os
    import datetime

    def get_latest_date(fname):
        date_strs = re.findall("[0-9]{2}-[0-9]{2}-[0-9]{4}", fname)
        tmp_dates = [datetime.datetime.strptime(i, "%m-%d-%Y") for i in date_strs]
        temp_date = max(tmp_dates)
        return temp_date

    # ===== Find the latest file to read
    comb_logfiles = [f for f in os.listdir(logfile_dir) if 'combined-stats.csv' in f]

    latest_date = get_latest_date(comb_logfiles[0])
    latest_file = comb_logfiles[0]

    for i in range(1, len(comb_logfiles)):
        fname = comb_logfiles[i]
        temp_date = get_latest_date(fname)
        if temp_date > latest_date:
            latest_date = temp_date
            latest_file = fname

    logger.info(f"Processing data from {latest_file} to build our DataFrame.")

    # ===== Read logfile
    logfile = os.path.join(logfile_dir, latest_file)
    loaded_df = pd.read_csv(logfile)

    return loaded_df


def build_shots_df(logfile_dir='srcdata/2019-2020_NBA_PbP_Logs', outfile='procdata/shots_df.csv', sm_df=False, overwrite=True):

    import os
    import sys

    logger.info("Building a DataFrame of all shots")

    logs_df = load_latest_logfile(logfile_dir)
    shots_df = process_shots_df(logs_df)
    shots_df.reset_index(inplace=True, drop=True)

    # ===== Write processed file
    if sm_df == True:
        shots_df = shots_df[[
            'date', 'period', 'away_score', 'home_score', 'remaining_time', 'elapsed', 'team', 'event_type',
            'assist', 'away', 'home', 'block', 'opponent', 'player',
            'shot_distance', 'original_x', 'original_y', 'shot_made', 'is_three', 'shot_zone', 'simple_zone'
        ]]
    shots_df = shots_df.assign(on_court=shots_df["a1"] + shots_df["a2"] + shots_df["a3"] + shots_df["a4"] + shots_df["a5"] + shots_df["h1"] + shots_df["h2"] + shots_df["h3"] + shots_df["h4"] + shots_df["h5"])

    if outfile is not None:
        if overwrite is not True:
            if os.path.exists(outfile):
                abort_bool = input(f"Output file {outfile} exists already - overwrite? 'y' for yes, otherwise no.")
                if abort_bool != "y":
                    sys.exit("Okay, exiting script.")
                else:
                    logger.info("Okay, proceeding with overwrite.")

        logger.info(f"Writing {outfile}...")

        shots_df.to_csv(outfile)

    return shots_df


def load_shots_df(shots_df_loc="procdata/shots_df.csv"):

    shots_df = pd.read_csv(shots_df_loc, index_col=0)

    return shots_df


def filter_shots_df(shots_df_in, teamname='NBA', period='All', player=None):

    avail_periods = ['All', 1, 2, 3, 4]

    if teamname == 'NBA':
        filtered_df = shots_df_in
    elif teamname in shots_df_in.team.unique():
        filtered_df = shots_df_in[shots_df_in.team == teamname]
    else:
        logger.error(f'{teamname} not in the list of teamnames! Returning input DF.')
        filtered_df = shots_df_in

    if period == 'All':
        filtered_df = filtered_df
    elif period in avail_periods:
        filtered_df = filtered_df[filtered_df.period == period]
    else:
        logger.error(f'{period} not in the list of possible periods! Returning input DF.')
        filtered_df = filtered_df

    if player is not None:
        filtered_df = filtered_df[filtered_df["player"] == player]

    return filtered_df


def get_pl_shot_counts(shots_df_in, crunchtime_mins=5):

    crunchtime_df = shots_df_in[
        (shots_df_in.tot_time > (48 - crunchtime_mins))
        & (shots_df_in.tot_time <= 48)
    ]
    pl_counts = crunchtime_df.groupby(['player', 'team']).count()['game_id'].sort_values()

    return pl_counts


def get_pl_data_dict(time_df, player, team, pl_acc_dict, pl_pps_dict, min_start, min_end, add_teamname=True):

    temp_df = time_df[time_df.player == player]
    team_temp_df = time_df[time_df.team == team]
    period = (min_start // 12) + 1

    if len(temp_df) == 0:
        shots_acc = 0
    else:
        shots_acc = sum(temp_df.shot_made) / len(temp_df)

    if len(team_temp_df) == 0:
        shots_freq = 0
    else:
        shots_freq = round(len(temp_df) / len(team_temp_df) * 100, 1)

    if add_teamname:
        player_name = player + ' [' + team + ']'
    else:
        player_name = player

    temp_dict = dict(
        player=player_name,
        pl_acc=round(pl_acc_dict[player][period] * 100, 1),  # overall accuracy
        pl_pps=round(pl_pps_dict[player][period] * 100, 1),  # overall PPS
        min_start=min_start + 1,
        min_mid=min_start + 0.5,
        min_end=min_end,
        shots_count=len(temp_df),
        shots_made=sum(temp_df.shot_made),
        shots_freq=shots_freq,
        shots_acc=round(100 * shots_acc, 1),  # acc for the sample only
        # shots_acc=shots_acc,
    )

    return temp_dict


def build_shot_dist_df(shots_df_loc="procdata/shots_df.csv", outfile='procdata/shot_dist_df.csv', overwrite=True):

    import os
    import sys

    logger.info("Building a DataFrame of shot distributions")

    shots_df = load_shots_df(shots_df_loc)

    # ========== PROCESS DATA FILE ==========
    shots_df.reset_index(inplace=True, drop=True)
    shots_df = shots_df[shots_df.period <= 4]

    # Get non-garbage time minutes:  Rule: up 13 with a minute left, increasing by one each minute
    shots_df = shots_df[shots_df["garbage"] == False]

    player_counts = get_pl_shot_counts(shots_df, crunchtime_mins=5)  # Sort by crunchtime shots, not just overall

    # Get top players for the summary list
    teams_list = set()
    top_players = list()
    for (player, pl_team) in player_counts.index[::-1]:
        if pl_team not in teams_list:
            top_players.append([player, pl_team])
            teams_list.add(pl_team)
    top_players = top_players[::-1]

    summary_data_list = list()

    # Get player data
    pl_acc_dict = dict()  # accuracies
    pl_pps_dict = dict()  # points per shot

    for player in shots_df.player.unique():
        pl_q_acc_dict = dict()
        pl_q_pps_dict = dict()
        for period in shots_df.period.unique():
            pl_df = shots_df[(shots_df.player == player) & (shots_df.period == period)]
            # Are there are any shots?
            if len(pl_df) > 0:
                pl_q_acc_dict[period] = sum(pl_df.shot_made) / len(pl_df.shot_made)
                pl_q_pps_dict[period] = (
                        (3 * sum(pl_df[pl_df.is_three].shot_made) + 2 * sum(pl_df[pl_df.is_three == False].shot_made))
                        / len(pl_df.shot_made)
                )
            else:
                pl_q_acc_dict[period] = 0
                pl_q_pps_dict[period] = 0

        pl_acc_dict[player] = pl_q_acc_dict
        pl_pps_dict[player] = pl_q_pps_dict

    # Set up range (number of minutes)
    min_range = 1

    for min_start in range(0, 48, min_range):
        min_end = min_start + min_range
        time_df = shots_df[(shots_df.tot_time > min_start) & (shots_df.tot_time <= min_end)]

        for player, team in top_players:
            pl_dict = get_pl_data_dict(time_df, player, team, pl_acc_dict, pl_pps_dict, min_start, min_end)
            summary_data_list.append(pl_dict)

    summary_df = pd.DataFrame(summary_data_list)
    summary_df = summary_df.assign(group="Leaders")

    part_thresh = 1  # Minimum % of team's shots to be shown on the chart
    # For each team:
    team_dfs = list()
    for team in shots_df.team.unique():

        team_df = shots_df[shots_df.team == team]
        player_counts = team_df.groupby('player').count()['game_id'].sort_values(ascending=True)

        # Consolidate non-qualifying players to 'Others'
        others_counts = player_counts[player_counts < sum(player_counts) / 100 * part_thresh]
        for temp_name in list(others_counts.index):
            team_df.player.replace(temp_name, 'Others', inplace=True)

        player_counts = get_pl_shot_counts(team_df, crunchtime_mins=5)  # Sort by crunchtime shots, not just overall

        # Get data for 'Others' as an aggreagate
        others_df = team_df[team_df.player == 'Others']
        pl_q_acc_dict = dict()
        pl_q_pps_dict = dict()
        for period in shots_df.period.unique():
            if len(others_df) > 0:
                pl_q_acc_dict[period] = sum(others_df.shot_made) / len(others_df.shot_made)
                pl_q_pps_dict[period] = (
                        (3 * sum(others_df[others_df.is_three].shot_made) + 2 * sum(others_df[others_df.is_three == False].shot_made))
                        / len(others_df.shot_made)
                )
            else:
                pl_q_acc_dict[period] = 0
                pl_q_pps_dict[period] = 0
        pl_acc_dict['Others'] = pl_q_acc_dict
        pl_pps_dict['Others'] = pl_q_pps_dict

        team_summary_data_list = list()
        min_range = 1

        for min_start in range(0, 48, min_range):
            min_end = min_start + min_range
            time_df = team_df[(team_df.tot_time > min_start) & (team_df.tot_time <= min_end)]

            for player, team in player_counts.index:
                pl_dict = get_pl_data_dict(time_df, player, team, pl_acc_dict, pl_pps_dict, min_start, min_end, add_teamname=False)
                team_summary_data_list.append(pl_dict)

        team_summary_df = pd.DataFrame(team_summary_data_list)
        team_summary_df = team_summary_df.assign(group=team)
        team_dfs.append(team_summary_df)

        # ===== END - COMPILE PLAYER DATA =====

    shot_dist_df = pd.concat(team_dfs + [summary_df])

    if outfile is not None:
        if overwrite is not True:
            if os.path.exists(outfile):
                abort_bool = input(f"Output file {outfile} exists already - overwrite? 'y' for yes, otherwise no.")
                if abort_bool != "y":
                    sys.exit("Okay, exiting script.")
                else:
                    logger.info("Okay, proceeding with overwrite.")

        logger.info(f"Writing {outfile}...")

        shot_dist_df.to_csv(outfile)

    return shot_dist_df


# def main():
#
# if __name__ == '__main__':
#      main()