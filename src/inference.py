import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def transcribe_audio(audio_path: str, model_dir: str) -> str:
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)

    audio, _ = librosa.load(audio_path, sr=16000)
    target_length = 16000 * 30
    audio = (
        np.pad(audio, (0, target_length - len(audio)))
        if len(audio) < target_length
        else audio[:target_length]
    )

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_features,
            language="en",
            task="transcribe",
            use_cache=False,
        )

    return processor.decode(outputs[0])


if __name__ == "__main__":
    MODEL_DIR = "./trained_model"
    AUDIO_PATH = "./test/sample.wav"

    result = transcribe_audio(AUDIO_PATH, MODEL_DIR)
    print(f"Classification: {result}")
