import locale
import sys
from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import resample

from base.config_loader import ConfigLoader
from base.data.dataloader import TuebingenDataloader


def alternate_signal_ww(signals, sample_left, sample_right):
    """ stretching/compressing of the signal + resampling"""
    # stretch/compress signal to the new window size using the sample to the left and right (for stretching)
    orig_size = signals.shape[0]
    new_size = int(ww_factor * orig_size)
    total_win = np.r_[sample_left, signals, sample_right]
    win_start = (total_win.shape[0] - new_size) // 2
    orig_win = total_win[win_start:win_start + new_size]

    # resample new signal to the old window size
    win = resample(orig_win, orig_size, axis=0)
    return win.astype('float32')


config = ConfigLoader('exp001', create_dirs=False)
config.DATA_FILE = 'D:/Python/mice_tuebingen/cache/dataset/data_tuebingen.h5'

sample = 10
ww_factor = 0.7
mapLoader = TuebingenDataloader(config, 'train', balanced=False, augment_data=False)
signal = mapLoader[sample][0].flatten()

mapLoader_augmented = TuebingenDataloader(config, 'train', balanced=False)
signal_augmented = alternate_signal_ww(signal, mapLoader[sample - 1][0].flatten()[:640],
                                       mapLoader[sample + 1][0].flatten()[-640:])

plt.rcParams.update({'font.size': 12})
locale.setlocale(locale.LC_NUMERIC, "de_DE")
plt.rcParams['axes.formatter.use_locale'] = True

fig, (ax1, ax2) = plt.subplots(2, sharex='all', figsize=(8, 4))

ax1.plot(np.arange(signal.shape[0]), signal, label='originales Signal', c='k')
ax1.axvspan((1 - ww_factor) / 2 * 1920, (1 - (1 - ww_factor) / 2) * 1920, alpha=0.3, color='darkgreen',
            label='neues Fenster')
ax1.legend()
ax1.set_ylabel('Amplitude')

ax2.plot(signal_augmented, label='transf. Signal', c='darkgreen')
ax2.legend()
ax2.set_ylabel('Amplitude')
ax2.set_ylim(ax1.get_ylim())

plt.xlabel('Fenstergröße in Datenpunkten')
plt.tight_layout()
plt.savefig(join(dirname(__file__), '../../..', 'results', 'plots', 'master', 'ww_example.svg'))
plt.show()
