# Non-Vocal Sound Classifier

A GPT/Whisper-based system for identification and transcription of non-vocal sounds, such as sirens, falling objects, collisions, vehicle engines, etc.

![Web interface](data/reports/figures/demo.gif)

## Description

This project is part of a research work aimed at developing a system capable of transforming non-vocal sounds into textual descriptions. We use a model based on OpenAI's Whisper, fine-tuned to identify different categories of environmental sounds.

## Project Structure


```
audio-classifier/
├── src/
│ ├── ml/ # Machine learning modules
| | ├── model/ # Whisper model
│ ├── backend/ # FastAPI API
│ └── frontend/ # Web interface (Flask)
├── data/
| ├── report/ # Project documentation
│ ├── sounds/ # Training data
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
# Method 1: Using the training script
python src/ml/train.py --audio_dir "data/sounds" --model_size "small" --epochs 10

# Method 2: Using the Jupyter notebook
jupyter notebook src/ml/train.ipynb
```

## Results

Some of the results obtained from the training process are shown below:

![Accuracy rate per epoch](data/reports/figures/fig1.png)
Figure 1: Accuracy rate per epoch

![Loss rate per epoch](data/reports/figures/fig2.png)
Figure 2: Loss rate per epoch
