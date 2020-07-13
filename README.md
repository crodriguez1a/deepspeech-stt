# deepspeech-stt


## Introduction

A slim Python interface for Mozilla's [DeepSpeech](https://github.com/mozilla/DeepSpeech/) STT

## Usage


```
from src.deepspeech_stt import deepspeech_predict

ouput_text: str = deepspeech_predict(wav_file_path)
```

See [notebook](notebooks/Examples.ipynb) for examples

## Installation

Download [Mozilla's DeepSpeech 0.7.4](https://github.com/mozilla/DeepSpeech/releases) pre-trained model (~200mb)

Then run:

```
poetry install
poetry shell
```
