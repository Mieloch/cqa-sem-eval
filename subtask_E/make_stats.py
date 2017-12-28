from argparse import ArgumentParser
from itertools import product
from stats import stats
from os import path

def main(verbose):
    data_files_dir = '/Volumes/DataDrive/stackexchange_train_v1_2'
    model_files_dir = '/Volumes/DataDrive/models'

    data_files = [
        data_files_dir + '/stackexchange_english_train_v1_2/stackexchange_english_train.xml',
        data_files_dir + '/stackexchange_english_train_v1_2/stackexchange_english_devel.xml',
        # data_files_dir + '/stackexchange_android_train_v1_2/stackexchange_android_train.xml',
        # data_files_dir + '/stackexchange_wordpress_train_v1_2/stackexchange_wordpress_train.xml',
        # data_files_dir + '/stackexchange_gaming_train_v1_2/stackexchange_gaming_train.xml',
    ]

    model_files = [
        '/Volumes/DataDrive/models/GoogleNews-vectors-negative300.bin',
        # model_files_dir + '/stackexchange_english_devel.xml--it50-mc5-s200.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it50-mc5-s100.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it10-mc5-s100.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it30-mc5-s100.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it10-mc10-s100.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it30-mc10-s100.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it50-mc10-s100.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it10-mc5-s200.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it30-mc5-s200.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it10-mc10-s200.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it30-mc10-s200.mdl',
        # model_files_dir + '/stackexchange_english_devel.xml--it50-mc10-s200.mdl',
    ]

    for data_file, model_file in product(data_files, model_files):
        dest_file_name = "/Volumes/DataDrive/stats/{}--{}.csv".format(
            path.basename(data_file),
            path.basename(model_file))

        if verbose:
            print('Writing to {}...'.format(dest_file_name))

        stats(data_file, dest=dest_file_name, model_path=model_file, verbose=verbose)

        if verbose:
            print('Done.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()
    main(args.verbose)
