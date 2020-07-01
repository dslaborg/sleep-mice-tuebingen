import numpy as np
from scipy.signal import resample

from base.config_loader import ConfigLoader


class DataAugmentor:
    def __init__(self, config: ConfigLoader):
        """ class for data augmentation, signals can be augmented using the method `alternate_signals` """
        self.config = config

    def alternate_signals(self, signals, sample_left, sample_right):
        """alternate signals in different ways, the augmentations to be applied can be configured in the config file

        the foolowing augmentations are implemented:
            - reinforcement of the amplitudes with a random factor in the range [1 - GAIN, 1 + GAIN]
            - flipping of single datapoints (datapoint * -1) if x~UNI(0,1) < FLIP (decision is made for every datapoint separately)
            - flipping of the whole signal (signal * -1) if x~UNI(0,1) < FLIP_ALL
            - horizontal flipping of the whole signal (signal is reversed) if x~UNI(0,1) < FLIP_HORI
            - stretching/compressing of the signal with a factor of x~UNI(1-WINDOW_WARP_SIZE, 1+WINDOW_WARP_SIZE) and resampling of this new signal to the original window size
            - shifting of the whole signal to the left or right by a factor of x~UNI(-TIME_SHIFT, TIME_SHIFT)

        Args:
            signals (np.ndarray): signal to be altered
            sample_left (np.ndarray): sample to the left of the signal, only relevant for window warping and time shift
            sample_right (np.ndarray): sample to the right of the signal, only relevant for window warping and time shift

        Returns:
            np.ndarray: augmented signal
        """
        if self.config.GAIN != 0:
            signals = self.alternate_signal_gain(signals)
        if self.config.FLIP != 0:
            signals = self.alternate_signal_flip(signals)
        if self.config.FLIP_ALL != 0:
            signals = self.alternate_signal_flip_all(signals)
        if self.config.WINDOW_WARP_SIZE != 0:
            signals = self.alternate_signal_ww(signals, sample_left, sample_right)
        if self.config.FLIP_HORI != 0:
            signals = self.alternate_signal_flip_hori(signals)
        if self.config.TIME_SHIFT != 0:
            signals = self.alternate_signal_time_shift(signals, sample_left, sample_right)
        return signals

    def alternate_signal_gain(self, signals):
        """ alter signals by multiplying them with a random factor (see `random_gain`) """
        return np.array([self.random_gain() * s for s in signals])

    def random_gain(self) -> float:
        """ random factor in [1 - GAIN, 1 + GAIN] """
        return 1.0 + self.config.GAIN * (2 * np.random.rand() - 1)

    def alternate_signal_flip(self, signals):
        """ flipping of single datapoints """
        return np.array([self.random_flip(signals.shape[1]) * s for s in signals], dtype='float32')

    def random_flip(self, n):
        return np.where(np.random.rand(n) < self.config.FLIP, -1, 1)

    def alternate_signal_flip_all(self, signals):
        """ flipping of whole signal """
        return np.array([self.random_flip_all() * s for s in signals], dtype='float32')

    def random_flip_all(self):
        return -1 if np.random.rand() < self.config.FLIP_ALL else 1

    def alternate_signal_ww(self, signals, sample_left, sample_right):
        """ stretching/compressing of the signal + resampling"""
        # stretch/compress signal to the new window size using the sample to the left and right (for stretching)
        orig_size = signals.shape[1]
        new_size = self.random_ww(orig_size)  # new window size
        total_win = np.c_[sample_left, signals, sample_right]
        win_start = (total_win.shape[1] - new_size) // 2
        orig_win = total_win[:, win_start:win_start + new_size]

        # resample new signal to the old window size
        win = resample(orig_win, orig_size, axis=1)
        return win.astype('float32')

    def random_ww(self, n):
        return int((1 + self.config.WINDOW_WARP_SIZE * (np.random.rand() * 2 - 1)) * n)

    def alternate_signal_flip_hori(self, signals):
        """ reversal of the signal """
        return np.fliplr(signals).copy() if self.random_flip_hori() else signals

    def random_flip_hori(self):
        return np.random.rand() < self.config.FLIP_HORI

    def alternate_signal_time_shift(self, signals, sample_left, sample_right):
        """ shifting of the signal to the left or right """
        orig_size = signals.shape[1]
        shift = self.random_time_shift(orig_size)  # number of datapoints to shift
        # concatenate signal with samples to the left and right and cut out shifted window
        total_win = np.c_[sample_left, signals, sample_right]
        win_start = sample_left.shape[1] + shift
        return total_win[:, win_start:win_start + orig_size].astype('float32')

    def random_time_shift(self, n):
        return int(self.config.TIME_SHIFT * (np.random.rand() * 2 - 1) * n)
