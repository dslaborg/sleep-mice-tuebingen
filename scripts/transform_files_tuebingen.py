import argparse
import sys
import time
from glob import glob
from os.path import join, basename, isfile, realpath, dirname

import h5py
import numpy as np
import tables

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.data.downsample import downsample
from base.config_loader import ConfigLoader
from base.data.data_table import create_table_description, COLUMN_LABEL, COLUMN_MOUSE_ID


def parse():
    parser = argparse.ArgumentParser(description='data transformation script')
    parser.add_argument('--experiment', '-e', required=True,
                        help='name of experiment to transform data to')

    return parser.parse_args()


def read_h5(h5_f_name):
    features = {}
    with h5py.File(config.DATA_DIR + h5_f_name, 'r') as h5_f:
        sampling_rates = [h5_f[c].attrs['samprate'][0] for c in config.CHANNELS]
        if np.all(sampling_rates == sampling_rates[0]):
            sr = sampling_rates[0]
        else:
            raise ValueError('sampling rates in original h5 files are not equal:', sampling_rates)

        for channel in config.CHANNELS:
            data_c = h5_f[channel][:]
            # Convert the integer data (2 bytes (int16)) into doubles
            scale_c = h5_f[channel].attrs['scale']
            offset_c = h5_f[channel].attrs['offset']
            features[channel] = np.double(data_c) * scale_c / 6553.6 + offset_c

            features[channel] = downsample(features[channel], sr_old=sr, sr_new=config.SAMPLING_RATE,
                                           fmax=0.4 * config.SAMPLING_RATE, outtype='sos', method='pad')

        scores = h5_f['Scoring/scores'][:]
        t_scoring_map = {v: k for k in config.SCORING_MAP for v in config.SCORING_MAP[k]}
        labels = [t_scoring_map[s] for s in scores]
        sample_start_times = h5_f['Scoring/times'][:]
        sample_start_times = (sample_start_times / sr * config.SAMPLING_RATE).astype('int')

    return features, labels, sample_start_times


def write_data_to_table(table, features, labels, start_times, mouse_id):
    sample = table.row

    for sample_start, label in zip(start_times, labels):
        sample_end = int(sample_start + config.SAMPLE_DURATION * config.SAMPLING_RATE)

        try:
            sample[COLUMN_MOUSE_ID] = mouse_id
            for c in config.CHANNELS:
                sample[c] = features[c][sample_start:sample_end]
            sample[COLUMN_LABEL] = config.STAGE_MAP[label]
            sample.append()
        except ValueError:
            print('not enough datapoints in sample, data from {} to {}'.format(sample_start, sample_end))
            print('total number of data points in file: {}'.format(len(list(features.values())[0])))
            print('this sample is ignored')
    table.flush()


def get_files_for_dataset(dataset):
    files = [basename(fn) for fn in glob(join(config.DATA_DIR, '*.h5'))]
    files.sort()

    # data
    data_split = config.DATA_SPLIT

    file_list = [f for mouse_name in data_split[dataset] for f in files if mouse_name in f]

    return file_list


def main():
    table_desc = create_table_description(config)

    if isfile(config.DATA_FILE):
        if input('{:s} already exists, do you want to override? (y/n)'.format(config.DATA_FILE)).lower() != 'y':
            exit()

    with tables.open_file(config.DATA_FILE, mode='w', title='data from Tuebingen') as f:

        datasets = ['train', 'valid', 'test']

        for dataset in datasets:
            print('writing', dataset, 'data...')

            table = f.create_table(f.root, dataset, table_desc, dataset + ' data')

            file_list = get_files_for_dataset(dataset)
            for i, f_name in enumerate(file_list):
                print('mouse [{:d}/{:d}]: {:s}'.format(i + 1, len(file_list), f_name))
                start = time.time()

                mouse_id = int(''.join(filter(str.isdigit, f_name)))
                features, labels, times = read_h5(f_name)
                write_data_to_table(table, features, labels, times, mouse_id)

                print('execution time: {:.2f}'.format(time.time() - start))
                print()

        print(f)


if __name__ == '__main__':
    args = parse()
    config = ConfigLoader(args.experiment)

    main()
