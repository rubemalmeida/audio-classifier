import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, audio_dir: str, processor):
        self.audio_paths = []
        self.labels = []
        self.processor = processor

        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith(".wav"):
                    self.audio_paths.append(os.path.join(root, file))
                    self.labels.append(os.path.basename(root))

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        audio, _ = librosa.load(audio_path, sr=16000)
        target_length = 16000 * 30
        audio = (
            np.pad(audio, (0, target_length - len(audio)))
            if len(audio) < target_length
            else audio[:target_length]
        )

        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )

        label_tokens = self.processor.tokenizer(
            label,
            padding="max_length",
            max_length=10,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_features": inputs.input_features[0],
            "labels": label_tokens.input_ids[0],
        }
