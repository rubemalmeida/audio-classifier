import torch
import numpy as np
import librosa
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.nn.functional import softmax


class SoundClassifier:
    def __init__(
        self,
        model_path,  # "openai/whisper-small" Changed to use the pretrained model directly
    ):
        # Check if model_path is a directory and doesn't have the required files
        if os.path.isdir(model_path) and not os.path.exists(
            os.path.join(model_path, "preprocessor_config.json")
        ):
            print(
                f"Warning: {model_path} does not contain required model files, using default Whisper model"
            )
            model_path = "openai/whisper-small"

        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.class_mapping = self._load_class_mapping()

    def _load_class_mapping(self):
        """Load mapping from class indices to class names"""
        # Hardcoded mapping matching the training data structure
        return {
            0: "ambulance",
            1: "firetruck",
            2: "traffic",
        }

    def classify(self, audio_path):
        """Classify the sound in the given audio file"""
        audio, sr = librosa.load(audio_path, sr=16000)
        target_length = 16000 * 30
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            # Access the encoder using get_encoder()
            encoder = self.model.get_encoder()
            encoder_outputs = encoder(inputs.input_features)
            logits = encoder_outputs.last_hidden_state.mean(dim=1)  # Aggregate embeddings

        # Apply softmax to get probabilities
        probabilities = softmax(logits, dim=-1).squeeze().cpu().numpy()
        probabilities2 = softmax(logits, dim=-1)
        data1, data2 = torch.topk(probabilities2,k=5)
        print(f"data1={data1}, data2={data2}")
        top_tokens = [self.processor.decode(idx) for idx in data2.squeeze().tolist()] # Decode top indices
        print(f"top_tokens={top_tokens}")

        # Map probabilities to classes
        class_probabilities = {
            class_name: probabilities[idx]
            for idx, class_name in self.class_mapping.items()
        }

        # Find the class with the highest probability
        class_match = max(class_probabilities, key=class_probabilities.get)
        confidence = class_probabilities[class_match]

        # Create a result format that matches the existing frontend expectations
        result = {
            "class": class_match,
            "confidence": confidence,
            "probabilities": class_probabilities,
        }

        return result
