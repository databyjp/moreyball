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


def main():
    shots_df = dataproc.build_shots_df(logfile_dir='srcdata/2019-2020_NBA_PbP_Logs', outfile='procdata/shots_df.csv', sm_df=False, overwrite=True)
    shot_dist_df = dataproc.build_shot_dist_df(shots_df_loc="procdata/shots_df.csv", outfile='procdata/shot_dist_df.csv', overwrite=True)


if __name__ == '__main__':
     main()