import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import os
import whisper

from src.ml.dataset import NonVocalSoundDataset
from src.ml.utils import save_model, get_device


def train(model_size="base", epochs=10, batch_size=16, learning_rate=0.0003):
    """Train a fine-tuned Whisper model for non-vocal sound classification"""

    device = get_device()
    print(f"Using device: {device}")

    # Load and initialize the pre-trained Whisper model
    model = whisper.load_model(model_size)

    # Modify the model architecture for our classification task
    num_classes = len(os.listdir("data/sounds"))  # Number of sound categories
    model.encoder.ln_post = nn.Identity()  # Remove the existing layer normalization
    model.encoder.add_module(
        "classifier",
        nn.Sequential(
            nn.Linear(model.encoder.n_audio_state, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        ),
    )
    model = model.to(device)

    # Define transformations
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
    )

    # Load datasets
    train_dataset = NonVocalSoundDataset(
        annotations_file="data/sounds/train_annotations.csv",
        audio_dir="data/sounds/train",
        transformation=mel_spectrogram,
        target_sample_rate=16000,
    )

    val_dataset = NonVocalSoundDataset(
        annotations_file="data/sounds/val_annotations.csv",
        audio_dir="data/sounds/val",
        transformation=mel_spectrogram,
        target_sample_rate=16000,
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model.encoder(data)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                outputs = model.encoder(data)
                loss = criterion(outputs, target)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    # Save the model
    save_model(model, f"whisper_{model_size}_nonvocal_classifier.pt")

    print("Training completed!")


if __name__ == "__main__":
    train()
