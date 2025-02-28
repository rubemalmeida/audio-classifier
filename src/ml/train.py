import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import whisper
import random
import numpy as np

from src.ml.utils import save_model, get_device


class DirectoryBasedSoundDataset(Dataset):
    def __init__(self, audio_dir, transformation, target_sample_rate=16000):
        self.audio_paths = []
        self.labels = []
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.class_mapping = {}

        # Vamos percorrer os diretórios e coletar os arquivos
        class_idx = 0
        for root, dirs, _ in os.walk(audio_dir):
            for class_dir in sorted(dirs):  # Ordenamos para garantir consistência
                class_path = os.path.join(root, class_dir)
                self.class_mapping[class_idx] = class_dir

                for file in os.listdir(class_path):
                    if file.endswith(".wav"):
                        self.audio_paths.append(os.path.join(class_path, file))
                        self.labels.append(class_idx)

                class_idx += 1

        print(
            f"Dataset criado com {len(self.audio_paths)} arquivos em {class_idx} classes"
        )
        print(f"Classes: {self.class_mapping}")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # Carrega o áudio com a biblioteca torchaudio para melhor integração com PyTorch
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample se necessário
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.target_sample_rate
                )
                waveform = resampler(waveform)

            # Converte para mono se estiver em estéreo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Aplicar a transformação (mel spectrogram)
            features = self.transformation(waveform)

            return features, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Erro ao carregar {audio_path}: {e}")
            # Retorna um tensor de zeros como fallback
            return torch.zeros((80, 3000)), torch.tensor(0, dtype=torch.long)


def train(
    audio_dir="data/sounds",
    model_size="base",
    epochs=10,
    batch_size=16,
    learning_rate=0.0003,
    val_split=0.2,
):
    """Train a fine-tuned Whisper model for non-vocal sound classification"""

    device = get_device()
    print(f"Using device: {device}")

    # Load and initialize the pre-trained Whisper model
    model = whisper.load_model(model_size)

    # Define transformations para mel spectrograms (padrão do Whisper)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
    )

    # Criando o dataset com base nos diretórios
    full_dataset = DirectoryBasedSoundDataset(
        audio_dir=audio_dir, transformation=mel_spectrogram, target_sample_rate=16000
    )

    # Determinar número de classes a partir do dataset
    num_classes = len(set(full_dataset.labels))

    # Modificar a arquitetura para classificação
    model.encoder.ln_post = nn.Identity()  # Remove a normalização existente
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

    # Dividir em treino e validação
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(val_split * dataset_size)
    train_indices = indices[split:]
    val_indices = indices[:split]

    # Criar os data loaders
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        full_dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler)

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

    # Salvar o mapeamento de classes
    with open(os.path.join("data/trained_model", "class_mapping.txt"), "w") as f:
        for idx, class_name in full_dataset.class_mapping.items():
            f.write(f"{idx},{class_name}\n")

    print("Training completed!")


if __name__ == "__main__":
    train()
