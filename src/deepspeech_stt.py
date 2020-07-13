import os
import shlex
import subprocess
import time
import wave

import librosa
import numpy as np
from deepspeech import Model
from scipy.io import wavfile as wav

from src.processing import denoise

try:
    from shhlex import quote
except ImportError:
    from pipes import quote


MODEL_PATH: str = os.getenv("MODEL_PATH", os.getcwd() + "/model/deepspeech-0.7.4-models.pbmm")


# Reference: https://deepspeech.readthedocs.io/en/v0.7.4/Python-Examples.html
def convert_samplerate(audio_path: str, desired_sample_rate: int) -> np.ndarray:
    sox_cmd: str = "sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither -".format(
        quote(audio_path), desired_sample_rate
    )
    output: bytes = b""
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("SoX returned non-zero status: {}".format(e.stderr))
    except OSError as e:
        raise OSError(
            e.errno,
            "SoX not found, use {}hz files or install it: {}".format(
                desired_sample_rate, e.strerror
            ),
        )

    return desired_sample_rate, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return "".join(token.text for token in metadata.tokens).strip()


def deepspeech_predict(wave_filename: str, noisy: bool = False) -> str:
    if not os.path.isfile(MODEL_PATH):
        raise Exception(f"Could not find model at {MODEL_PATH}")

    ds: Model = Model(MODEL_PATH)
    fs, audio = convert_samplerate(wave_filename, ds.sampleRate())

    # NOTE: experimental
    if noisy:
        audio = denoise(audio.astype("float32"), ds.sampleRate())
        audio = audio.astype("int16")

    return metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0])


if __name__ == "__main__":
    tic = time.perf_counter()
    predicted_text = deepspeech_predict(os.getenv("SAMPLE_WAVE", "samples/test.wav"))
    toc = time.perf_counter()
    print("---")
    print(predicted_text)
    print("---")
    print(f"Time elapsed was {toc - tic:0.4f} seconds")
