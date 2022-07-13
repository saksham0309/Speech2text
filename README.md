# Speech-to-Text (S2T) Modeling

Speech recognition (ASR) and speech-to-text translation (ST) with fairseq.

## Dependencies
```
PyTorch
Tensorflow
Argparse
Numpy
Transformer
Tokenizer
```

## Data Preparation
S2T modeling data consists of source speech features, target text and other optional information
(source text, speaker id, etc.). Fairseq S2T uses per-dataset-split TSV manifest files
to store these information. Each data field is represented by a column in the TSV file.

Unlike text token embeddings, speech features (e.g. log mel-scale filter banks) are usually fixed
during model training and can be pre-computed. The manifest file contains the path to
either the feature file in NumPy format or the WAV/FLAC audio file. For the latter,
features will be extracted on-the-fly by fairseq S2T. Optionally, feature/audio files can be packed
into uncompressed ZIP files (then accessed via byte offset and length) to improve I/O performance.

Fairseq S2T also employs a YAML file for data related configurations: tokenizer type and dictionary path
for the target text, feature transforms such as CMVN (cepstral mean and variance normalization) and SpecAugment,
temperature-based resampling, etc.

## Model Training
Fairseq S2T uses the unified `fairseq-train` interface for model training. It requires arguments `--task speech_to_text`,
 `--arch <model architecture in fairseq.models.speech_to_text.*>` and `--config-yaml <config YAML filename>`.

## Inference & Evaluation
Fairseq S2T uses the unified `fairseq-generate`/`fairseq-interactive` interface for inference and evaluation. It
requires arguments `--task speech_to_text` and `--config-yaml <config YAML filename>`. The interactive console takes
audio paths (one per line) as inputs.


## Examples
- [Speech Recognition (ASR) on LibriSpeech](docs/librispeech_example.md)

- [Speech-to-Text Translation (ST) on MuST-C](docs/mustc_example.md)

- [Speech-to-Text Translation (ST) on CoVoST 2](docs/covost_example.md)

- [Speech-to-Text Translation (ST) on Multilingual TEDx](docs/mtedx_example.md)
- [Simultaneous Speech-to-Text Translation (SimulST) on MuST-C](docs/simulst_mustc_example.md)


