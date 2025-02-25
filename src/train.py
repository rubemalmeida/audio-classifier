import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import time
from dataset import AudioDataset


def train_model(audio_dir: str, output_dir: str, num_epochs: int = 10):
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    dataset = AudioDataset(audio_dir, processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0

        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
        print(f"Epoch time: {time.time() - start_time:.2f} seconds")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    AUDIO_DIR = "sounds"
    OUTPUT_DIR = "trained_model"
    NUM_EPOCHS = 3

    train_model(AUDIO_DIR, OUTPUT_DIR, NUM_EPOCHS)
