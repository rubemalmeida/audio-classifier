import os
import torch
import torchaudio
import numpy as np


def get_device():
    """Get the available device (CUDA or CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, filename):
    """Save the trained model"""
    os.makedirs("data/trained_model", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("data/trained_model", filename))
    print(f"Model saved to data/trained_model/{filename}")


def load_model(model_path):
    """Load a trained model"""
    device = get_device()
    model = torch.load(model_path, map_location=device)
    return model


def load_audio(file_path, target_sr=16000, max_duration=30):
    """Load audio file and preprocess for the model"""
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)

    # Trim to max_duration seconds
    max_samples = target_sr * max_duration
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # Convert to numpy array
    waveform = waveform.squeeze().numpy()

    return waveform


def preprocess_audio(file_path, target_sr=16000, max_duration=30):
    """Preprocess audio for Whisper-based model"""
    audio = load_audio(file_path, target_sr, max_duration)

    # Apply any necessary preprocessing for the model
    # For example, normalize the audio
    audio = audio / np.max(np.abs(audio))

    return audio
