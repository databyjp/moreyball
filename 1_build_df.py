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

import dataproc
import pandas as pd
import numpy as np
import os
import re


def main():

    logdir = 'srcdata'
    logfiles = [f for f in os.listdir(logdir) if 'combined-stats.csv' in f]
    for logfile in logfiles[:]:

        logger.info(f"Loading {logfile}")
        season_dates = re.findall(r"[0-9]{2}-[0-9]{2}-[0-9]{4}", logfile)
        season_yrs = [i[-2:] for i in season_dates]

        log_df = None
        try:
            log_df = pd.read_csv(os.path.join(logdir, logfile))
        except UnicodeDecodeError:
            logger.warning("Could not load logfile, trying with ISO-8859-1 encoding")
            try:
                log_df = pd.read_csv(os.path.join(logdir, logfile), encoding="ISO-8859-1")
            except UnicodeDecodeError:
                logger.exception(f"Sorry! Couldn't load {logfile}")

        if log_df is not None:
            outdir = "procdata"
            out_df_path = os.path.join(outdir, f'shots_df_{season_yrs[0]}_{season_yrs[1]}.csv')
            logger.info(f"Loaded log_df with {len(log_df)} lines")
            try:
                shots_df = dataproc.build_shots_df(log_df, outfile=out_df_path, sm_df=False, overwrite=True)
                logger.info(f"Saved shots_df with {len(shots_df)} lines")
                logger.info(f"{len(shots_df)} shots processed for {logfile}")
            except:
                logger.error(f"There was a problem converting {out_df_path}")

            # out_dist_df_path = os.path.join(outdir, f'shot_dist_df_{season_yrs[0]}_{season_yrs[1]}.csv')
            # shot_dist_df = dataproc.build_shot_dist_df(shots_df, outfile=out_dist_df_path, overwrite=True)


if __name__ == '__main__':
     main()