# Audio to Text Classifier

A deep learning model based on Whisper that converts non-speech audio into textual descriptions.

## Overview

This project uses OpenAI's Whisper model, fine-tuned to classify and describe non-speech audio signals into text. It can be used for various applications such as environmental sound classification, machine sound analysis, and audio event detection.

## Installation

```bash
git clone https://github.com/rubemalmeida/audio-classifier
cd audio-classifier
pip install -r requirements.txt
```

## Usage

### Training

Place your audio files in folders named after their categories under the `sounds` directory:

```
sounds/
├── categoryX/
│ ├── sound1.wav
│ └── sound2.wav
└── category2/
  ├── sound3.wav
  └── sound4.wav
```

Then run:

```bash
python src/train.py
```

### Inference

```bash
python src/infer.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See requirements.txt for full list

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.