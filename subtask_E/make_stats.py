"""
Script to create multiple stats .csv files.
"""

from argparse import ArgumentParser
from itertools import product
from collections import namedtuple
from stats import stats


StatsJob = namedtuple('StatsJob', ['name', 'data', 'model'])


def main(verbose):
    jobs = [
        StatsJob(name='en_train_stats.csv',
                 data='/Volumes/DataDrive/stackexchange_train_v1_2/stackexchange_english_train_v1_2/stackexchange_english_train.xml',
                 model='/Volumes/DataDrive/models/GoogleNews-vectors-negative300.bin'),
        StatsJob(name='en_test_stats.csv',
                 data='/Volumes/DataDrive/stackexchange_train_v1_2/stackexchange_english_train_v1_2/stackexchange_english_devel.xml',
                 model='/Volumes/DataDrive/models/GoogleNews-vectors-negative300.bin'),
    ]

    for job in jobs:
        dest_file_name = "/Volumes/DataDrive/stats/{}".format(job.name)

        if verbose:
            print('Writing to {}...'.format(dest_file_name))

        stats(job.data, dest=dest_file_name,
              model_path=job.model, verbose=verbose)

        if verbose:
            print('Done.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()
    main(args.verbose)
