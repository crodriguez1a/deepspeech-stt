import librosa  # https://github.com/librosa/librosa/issues/1160
import numpy as np
import scipy


def apply_filter(S: np.ndarray, sr: int) -> tuple:
    """
    Ref:
    https://librosa.github.io/librosa/0.7.0/auto_examples/plot_vocal_separation.html#vocal-separation
    Brian McFee, ISC
    """
    S_filter = librosa.decompose.nn_filter(
        S,
        aggregate=np.median,
        metric="cosine",
        width=int(librosa.time_to_frames(2, sr=sr)),
    )

    S_filter = np.minimum(S, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter, margin_i * (S - S_filter), power=power)
    mask_v = librosa.util.softmask(S - S_filter, margin_v * S_filter, power=power)
    return (mask_i, mask_v)


def denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    print("Spectral representation")
    S: np.ndarray = np.abs(librosa.stft(audio))  # mel_features(audio, sample_rate)

    print("Filtering away background")
    fg_mask, bg_mask = apply_filter(S, sample_rate)
    foreground: np.ndarray = fg_mask * S

    print("Inversing to audio")
    return librosa.istft(foreground, length=len(audio))
