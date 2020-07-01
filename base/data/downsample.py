from typing import Tuple
from functools import lru_cache

import numpy as np

from scipy.signal import butter as _butter
from scipy.signal import filtfilt as _filtfilt
from scipy.signal import sosfiltfilt as _sosfiltfilt


def _highpass(sr: float, f: float, outtype='sos') -> Tuple[np.ndarray, np.ndarray]:
    highpass_freq = 2 * f / sr
    return _butter(4, highpass_freq, btype='high', output=outtype)


def _lowpass(sr: float, f: float, outtype='sos') -> Tuple[np.ndarray, np.ndarray]:
    lowpass_freq = 2 * f / sr
    return _butter(2, lowpass_freq, btype='low', output=outtype)


def _bandpass(sr: float, band: Tuple[float, float], outtype='sos') -> Tuple[np.ndarray, np.ndarray]:
    bandpass_freqs = 2 * np.array(band) / sr
    return _butter(4, bandpass_freqs, btype='band', output=outtype)


@lru_cache(maxsize=128)
def _pass(sr: float, band: Tuple[float, float], outtype='sos') -> Tuple[np.ndarray, np.ndarray]:
    try:
        assert not all(f is None for f in band), "fmin and fmax is `None`."
        if band[1] is None:
            return _highpass(sr, band[0], outtype)
        elif band[0] is None:
            return _lowpass(sr, band[1], outtype)
        else:
            assert band[0] < band[1], "fmin>=fmax (fmin: %s, fmax: %s)" % band
            return _bandpass(sr, band, outtype)
    except AssertionError as err:
        raise ValueError(str(err))


def filtfilt(x, sr, fmin=None, fmax=None, axis=-1, outtype='sos', method='pad'):
    """ applies filter based on `outtype` """
    if outtype == 'sos':
        sos = _pass(sr, (fmin, fmax), outtype)
        return _sosfiltfilt(sos, x, axis=axis, padtype='constant')
    elif outtype == 'ba':
        b, a = _pass(sr, (fmin, fmax), outtype)
        return _filtfilt(b, a, x, axis=axis, padtype='constant', method=method)
    else:
        raise ValueError('outtype neither sos nor ba')


def downsample(x, sr_old, sr_new, fmin=None, fmax=None, outtype='sos', method='pad'):
    """resample `x` linearly with sampling rate `sr_new`

    Applies 4-th order Butterworth low-pass filter with `fmax` to `x`. Then
    resamples linearly to `sr_new`. Resampled times start at `t[0]=0` and ends
    at `t[-1]=np.round(x.size*sr_new/sr_old)/sr_new`. If
    `sr_new==sr_old`, resampling is not performed, but the filter is still
    applied.

    Args:
        x (np.ndarray): samples to be down-sampled.
        sr_old (float): sampling rate in Hz of `x`.
        sr_new (float): sampling rate in Hz to which `x` shall be resampled.
        fmin (float or None): high-pass edge of the filter to be applied to `x` before down-sampling.
        fmax (float or None): low-pass edge of the filter to be applied to `x` before down-sampling.
        outtype (str): 'sos' or 'ba', type of filter to be used, scipy.signal.filtfilt or scipy.signal.sosfiltfilt
        method (str): 'pad' or 'gust', only used if method is 'ba', for more details see scipy.signal.filtfilt

    Returns:
        np.ndarray: downsampled signal
    """
    # Filter before down-sampling
    try:
        assert sr_old >= sr_new, """new sampling rate larger than old rate (sr_old: %s, sr_new: %s)""" % (
            sr_old, sr_new)
        assert fmax <= 0.4 * sr_new, """fmax %s > 0.8*f_nyquist of new sampling rate""" % fmax
    except AssertionError as err:
        raise ValueError(str(err))
    x = filtfilt(x, sr=sr_old, fmin=fmin, fmax=fmax, outtype=outtype, method=method)

    # Resample to new sampling rate
    if not sr_old == sr_new:
        t_old = np.linspace(0, x.size / sr_old, x.size)
        num_samples_new = np.round(x.size / sr_old * sr_new).astype('int')
        t_new = np.linspace(0, num_samples_new / sr_new, num_samples_new)
        x = np.interp(t_new, t_old, x)
    return x
