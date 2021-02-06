#!/usr/bin/env python

import sys
from glob import glob
from os.path import join, basename, realpath, dirname

import h5py
import numpy as np

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.config_loader import ConfigLoader
from base.evaluation.matrix_plotter import MatrixPlotter

# Wake: 0
# REM: 1
# Non REM: 2
# Pre REM: 3

# original
# scoring_map = {1: 'Wake', 2: 'Non REM', 3: 'REM', 4: 'Pre REM', 8: 'artifact', 17: 'Wake', 18: 'Non REM', 19: 'REM',
#                20: 'Pre REM', 24: 'artifact'}

# original w/o artifacts
orig_scoring_map = {1: 0, 2: 2, 3: 1, 4: 3, 17: 0, 18: 2, 19: 1, 20: 3}

# pre REM -> non REM
# artifacts are ignored
scoring_map = {1: 0, 2: 2, 3: 1, 4: 2, 17: 0, 18: 2, 19: 1, 20: 2}


def read_h5(h5_f_name: str):
    # open h5 file in DATA_DIR
    with h5py.File(join(config.DATA_DIR, h5_f_name), 'r') as h5_f:
        # load scores from h5 file and map them to stages using SCORING_MAP
        scores = h5_f['Scoring/scores'][:]
        orig_labels = [orig_scoring_map[s] for s in scores if s in orig_scoring_map]
        proc_labels = [scoring_map[s] for s in scores if s in scoring_map]

        def rem_to_pre_rem(lab, p_lab1, p_lab2):
            return lab != 1 \
                   and (p_lab1 == 1 or p_lab2 == 1)

        proc_labels = [3 if i + 2 < len(proc_labels) and rem_to_pre_rem(*proc_labels[i:i + 3]) else proc_labels[i] for i
                       in range(len(proc_labels))]
        return orig_labels, proc_labels


def get_files_for_dataset():
    """determines which files to transform for a given dataset, see DATA_SPLIT"""
    # load all h5 files in DATA_DIR and sort them
    files = [basename(fn) for fn in glob(join(config.DATA_DIR, '*.h5'))]
    files.sort()

    return files


def transform():
    """transform files in DATA_DIR to pytables table"""

    print(f'data is loaded from {realpath(config.DATA_DIR)}')

    files = get_files_for_dataset()
    combined_orig_labels = []
    combined_proc_labels = []
    matrix_plotter = MatrixPlotter(config)

    # iterate over files, load them and write them to the created table
    for i, f_name in enumerate(files):
        print('mouse [{:d}/{:d}]: {:s}'.format(i + 1, len(files), f_name))
        # read h5 file
        orig_labels, proc_labels = read_h5(f_name)
        combined_orig_labels.extend(orig_labels)
        combined_proc_labels.extend(proc_labels)

    tm = matrix_plotter.plot_tm(combined_proc_labels, wo_plot=True)
    print([[float(f'{j * 100:.2f}') for j in i] for i in tm])

    cm = matrix_plotter.plot_cm(combined_orig_labels, combined_proc_labels,
                                np.array(['Wake', 'REM', 'NREM', 'pre-REM']), normalize=1, wo_plot=True)
    print([[float(f'{j * 100:.2f}') for j in i] for i in cm])


if __name__ == '__main__':
    # load config, dirs are not needed here, because we do not write a log
    config = ConfigLoader(create_dirs=False)

    # transform files
    transform()
