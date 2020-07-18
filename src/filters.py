import numpy as np
from numpy import pi, sin
from pydub import AudioSegment
from scipy.signal import butter, lfilter


def _butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 50.0,
    highcut: float = 3000.0,
    fs: int = 16000,
    order: int = 5,
):
    # Ref: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    # Ref: https://dsp.stackexchange.com/questions/2993/human-speech-noise-filter
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def normalize(data: np.ndarray):
    # Ref: https://github.com/ipython/ipython/blob/f8c9ea7db42d9830f16318a4ceca0ac1c3688697/IPython/lib/display.py
    data = np.array(data, dtype=float)
    if len(data.shape) == 2:
        # In wave files,channels are interleaved. E.g.,
        # "L1R1L2R2..." for stereo. See
        # http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
        data = data.T.ravel()
    else:
        raise ValueError("Array audio input must be a 1D or 2D array")

    max_abs_value = np.mean(np.abs(data))
    scaled = data / max_abs_value * 32767
    return scaled


def dub_segment(data: np.ndarray, sample_rate: int = 16000):
    return AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=1,
    )


def low_pass_filter(
    data: np.ndarray, sample_rate: int = 16000, cuttoff=500
) -> np.ndarray:
    # Ref: https://github.com/jiaaro/pydub/blob/master/pydub/scipy_effects.py
    audio_segment = dub_segment(data, sample_rate)
    return np.array(audio_segment.low_pass_filter(cuttoff).get_array_of_samples())


def high_pass_filter(
    data: np.ndarray, sample_rate: int = 16000, cuttoff=500
) -> np.ndarray:
    # Ref: https://github.com/jiaaro/pydub/blob/master/pydub/scipy_effects.py
    audio_segment = dub_segment(data, sample_rate)
    return np.array(audio_segment.high_pass_filter(cuttoff).get_array_of_samples())
