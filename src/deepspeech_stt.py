import os
import shlex
import subprocess
import time
from typing import Any, List, Sequence

import librosa
import numpy as np
import scipy.signal as signal
from deepspeech import CandidateTranscript, Model
from scipy.io import wavfile as wav
from tqdm.auto import tqdm

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

from src.filters import (
    butter_bandpass_filter,
    high_pass_filter,
    low_pass_filter,
    logmmse_denoise,
)

MODEL_PATH: str = os.getenv(
    "MODEL_PATH", os.getcwd() + "/model/deepspeech-0.7.4-models.pbmm"
)

SIGNAL_FILTERS: list = [
    (high_pass_filter, ()),
    (low_pass_filter, ()),
    (butter_bandpass_filter, ()),
    (logmmse_denoise, ([16000])),
]


def convert_samplerate(audio_path: str, desired_sample_rate: int) -> np.ndarray:
    """
    Apply sample rate conversion

    # Ref: https://deepspeech.readthedocs.io/en/v0.7.4/Python-Examples.html
    """
    sox_cmd: str = """
    sox {} --type raw\
    --bits 16 --channels 1 --rate {}\
    --encoding signed-integer --endian little\
    --compression 0.0 --no-dither -
    """.format(
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


def metadata_to_string(metadata: CandidateTranscript):
    """
    Translate CandidateTranscript to plain text
    """
    return "".join(token.text for token in metadata.tokens).strip()


def apply_filters(audio: np.ndarray, filters: list):
    for filter in filters:
        fn = [(f, params) for f, params in SIGNAL_FILTERS if f.__qualname__ == filter]
        if fn:
            filt, params = fn[0]
            audio = filt(audio, *params).astype("int16")
    return audio


def batching_after_silence(
    audio: np.ndarray,
    silence_threshold: int,
    model: Model,
    verbose: bool = False,
    filters: list = None,
) -> List[Any]:
    """
    Infer after natural gaps of silence

    Ref: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    """
    results: list = []
    audio = audio.astype("float32")
    y: np.ndarray = librosa.effects.split(audio, top_db=silence_threshold, ref=np.mean)

    clips: list = []
    for i in tqdm(y):
        clip = audio[i[0] : i[1]]
        clip = clip.astype("int16")

        if filters:
            clip = apply_filters(clip, filters)

        clips.append((clip, filters or ["no filter"]))

    for clip, meta in tqdm(clips):
        transcripts = metadata_to_string(model.sttWithMetadata(clip, 1).transcripts[0])
        if transcripts and verbose:
            print(transcripts, " : ", meta)
        results.append(transcripts)

    return results


def deepspeech_predict(
    wave_filename: str,
    batch_after_silence: bool = False,
    silence_threshold: int = 50,
    verbose: bool = False,
    filters: list = None,
) -> Sequence[Any]:
    """
    DeepSpeech is an open source Speech-To-Text engine, using a model trained
    by machine learning techniques based on Baidu’s Deep Speech research paper.
    """
    if not os.path.isfile(MODEL_PATH):
        raise Exception(
            f"""
        Could not find model at {MODEL_PATH}.

        Download Model from:
        https://github.com/mozilla/DeepSpeech/releases

        Export environment variable:
        `export MODEL_PATH=/path/to/model/`
        """
        )

    model: Model = Model(MODEL_PATH)
    sample_rate: int = model.sampleRate()
    fs, audio = convert_samplerate(wave_filename, sample_rate)

    if batch_after_silence:
        results = batching_after_silence(audio, silence_threshold, model, verbose, filters)
        return results

    elif filters:
        audio = apply_filters(audio, filters)

    return metadata_to_string(model.sttWithMetadata(audio, 1).transcripts[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path of wav")
    parser.add_argument("--silence_threshold", help="Threshold for silence", default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--batch_after_silence",
        action="store_true",
        help="Returns sequence of transcriptions inferred after natural silence",
    )
    parser.add_argument("--filters", help="Signal filter", nargs="+")
    args = parser.parse_args()

    tic = time.perf_counter()
    predicted_text: Sequence[Any] = deepspeech_predict(
        args.path,
        silence_threshold=int(args.silence_threshold),
        verbose=args.verbose,
        batch_after_silence=args.batch_after_silence,
        filters=args.filters,
    )

    print("---")
    print(predicted_text)
    toc = time.perf_counter()
    print(f"Inference time elapsed was {toc - tic:0.4f} seconds")
