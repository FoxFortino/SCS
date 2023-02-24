"""
This script will generate a pandas dataframe from two lists of lnw files.

This script is designed to be run from the command line. You can modify the
hardcoded data directories and save directories. The defaults were appropriate
for my file system but won't be for you. This script only needs to be run
once.

However, the result of this file `sn_data.parquet` should already exist in the
directory data directory of this package.
"""
from os import getcwd
from os.path import join
from os.path import abspath
from glob import glob

import data_loading as dl


def main(save_dir, data1_lnws, data2_lnws):
    no_duplicate_lnw = dl.remove_duplicate_lnw(data1_lnws, data2_lnws)
    no_bad_lnw = dl.remove_bad_sn(no_duplicate_lnw)
    sn_data0 = dl.load_lnws(no_bad_lnw, save_dir)

    sn_data_file = join(save_dir, "sn_data.parquet")
    print(f"Dataframe saved to: {sn_data_file}")


if __name__ == "__main__":
    DASH_data_dir = "/home/2649/repos/adfox/templates/training_set"
    DASH_data_lnws = glob(join(DASH_data_dir, "*lnw"))

    SESN_data_dir = "/home/2649/repos/SESNtemple/SNIDtemplates"
    SESN_data_lnws = glob(join(SESN_data_dir, "templates_*/*lnw"))

    save_dir = abspath(join(getcwd(), "../data"))

    main(save_dir, DASH_data_lnws, SESN_data_lnws)
