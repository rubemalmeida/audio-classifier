# Non-Vocal Sound Classifier

A GPT/Whisper-based system for identification and transcription of non-vocal sounds, such as sirens, falling objects, collisions, vehicle engines, etc.

## Description

This project is part of a research work aimed at developing a system capable of transforming non-vocal sounds into textual descriptions. We use a model based on OpenAI's Whisper, fine-tuned to identify different categories of environmental sounds.

## Project Structure


```
audio-classifier/
├── src/
│ ├── ml/ # Machine learning modules
│ ├── backend/ # FastAPI API
│ └── frontend/ # Web interface (Flask)
├── data/
│ ├── sounds/ # Training data
│ └── trained_model/ # Saved models
└── reports/ 
```

## Features

- Web interface for uploading or recording audio
- REST API for processing and classifying audio
- Support for .wav files
- Automatic processing to 16kHz frequency
- 30-second limit per audio
- Model trained to identify various categories of non-vocal sounds

## Requirements

- Python 3.8+
- PyTorch
- Whisper
- FastAPI
- Flask
- Other dependencies specified in requirements.txt

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the backend: `python -m src.backend.main`
4. Run the frontend: `python -m src.frontend.app`

## Usage

1. Access the web interface at http://localhost:5000
2. Upload an audio file or record a new one
3. Click on "Classify Sound"
4. View the classification results

## Model Training

To train a new model:

```bash
cd src/ml
python train.py
```

