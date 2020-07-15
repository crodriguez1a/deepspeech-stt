import os
import shlex
import subprocess
import time
from typing import Any, List, Sequence

import librosa
import numpy as np
from deepspeech import CandidateTranscript, Model
from logmmse import logmmse
from scipy.io import wavfile as wav
from tqdm.auto import tqdm

try:
    from shhlex import quote
except ImportError:
    from pipes import quote


MODEL_PATH: str = os.getenv(
    "MODEL_PATH", os.getcwd() + "/model/deepspeech-0.7.4-models.pbmm"
)


def convert_samplerate(audio_path: str, desired_sample_rate: int) -> np.ndarray:
    """
    Apply sample rate conversion

    # Ref: https://deepspeech.readthedocs.io/en/v0.7.4/Python-Examples.html
    """
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


def metadata_to_string(metadata: CandidateTranscript):
    """
    Translate CandidateTranscript to plain text
    """
    return "".join(token.text for token in metadata.tokens).strip()


def logmmse_denoise(audio: np.ndarray, sr: int):
    """
    LogMMSE speech enhancement/noise reduction algorithm
    """
    return logmmse(audio, sr)


def batch_on_silence(
    audio: np.ndarray, top_db: int, model: Model, verbose: bool
) -> List[Any]:
    """
    Infer after natural gaps of silence

    Ref: http://jamesmontgomery.us/blog/Voice_Recognition_Model.html
    """
    results: list = []
    audio = audio.astype("float32")
    y = librosa.effects.split(audio, top_db=top_db, ref=np.mean)
    for i in tqdm(y):
        clip = audio[i[0] : i[1]]
        clip = clip.astype("int16")
        transcripts = metadata_to_string(model.sttWithMetadata(clip, 1).transcripts[0])
        if transcripts and verbose:
            print(transcripts, "...")
        results.append(transcripts)

    return results


def deepspeech_predict(
    wave_filename: str,
    infer_after_silence: bool = True,
    top_db: int = 50,
    output_sentences: bool = False,
    verbose: bool = False,
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

    if infer_after_silence:
        results = batch_on_silence(audio, top_db, model, verbose)
        return results if output_sentences else " ".join(results)

    return metadata_to_string(model.sttWithMetadata(audio, 1).transcripts[0])


if __name__ == "__main__":
    tic = time.perf_counter()
    predicted_text = deepspeech_predict(os.getenv("SAMPLE_WAVE", "samples/test.wav"))
    toc = time.perf_counter()
    print("---")
    print(predicted_text)
    print("---")
    print(f"Time elapsed was {toc - tic:0.4f} seconds")
