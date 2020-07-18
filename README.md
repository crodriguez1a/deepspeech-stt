# deepspeech-stt


## Introduction

A slim Python client for Mozilla's [DeepSpeech](https://github.com/mozilla/DeepSpeech/) speech-to-text

## Usage


```
from src.deepspeech_stt import deepspeech_predict

ouput_text: str = deepspeech_predict(wav_file_path)
```

Parameter | Default | Description
---|---|---
`wave_filename` | `None` | Path to wave file
`infer_after_silence`|`True`| Batch predictions split by natural gaps of silence
`silence_threshold` | `50` | The threshold (in decibels) below<br>reference to consider as silence
`filter`| `None` | Choice of signal filtering:<br> `butter_bandpass_filter`, `high_pass_filter`, `low_pass_filter`
See [notebook](notebooks/Examples.ipynb) for examples

## Installation

Download [Mozilla's DeepSpeech 0.7.4](https://github.com/mozilla/DeepSpeech/releases) pre-trained model (~200mb)

Then run:

```
poetry install
poetry shell
```
