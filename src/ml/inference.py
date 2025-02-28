import torch
import torchaudio
import whisper
import numpy as np
import torch.nn as nn
from src.ml.utils import get_device, load_audio, load_model


class SoundClassifier:
    def __init__(
        self,
        model_path="data/trained_model/whisper_base_nonvocal_classifier.pt",
        model_size="base",
    ):
        self.device = get_device()
        # Initialize class_mapping first
        self.class_mapping = self._load_class_mapping()
        # Then load the model which uses self.class_mapping
        self.model = self._load_model(model_path, model_size)
        self.model.eval()

    def _load_model(self, model_path, model_size):
        """Load the fine-tuned model"""
        base_model = whisper.load_model(model_size)

        # Inspect model structure more thoroughly
        print("Model structure info:")
        print(f"Model dimensions: {base_model.dims}")

        # Get class count from class_mapping
        num_classes = len(self.class_mapping)

        # Get the correct dimension from the Whisper model
        # The Whisper model has a 'dims' attribute that contains all dimensions
        # Typically, we should use encoder_dim = base_model.dims.n_audio_state
        encoder_dim = base_model.dims.n_audio_state
        print(f"Using encoder dimension: {encoder_dim}")

        # Modify model architecture to match training architecture
        base_model.encoder.ln_post = nn.Identity()
        base_model.encoder.add_module(
            "classifier",
            nn.Sequential(
                nn.Linear(encoder_dim, 512),  # Using the correct dimension
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            ),
        )

        # Load trained weights
        try:
            base_model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(
                f"Warning: Model file not found at {model_path}. Using base model only."
            )
        except Exception as e:
            print(
                f"Warning: Could not load model weights: {str(e)}. Using base model only."
            )

        base_model = base_model.to(self.device)
        return base_model

    def _load_class_mapping(self):
        """Load mapping from class indices to class names"""
        # This would typically load from a file saved during training
        # For example purposes, hardcoding a sample mapping
        return {
            0: "sirene",
            1: "queda_de_objeto",
            2: "colisao_de_objetos",
            3: "motor_de_veiculo",
            4: "buzina",
            5: "vidro_quebrando",
            # Add more classes as needed
        }

    def classify(self, audio_path):
        """Classify the sound in the given audio file"""
        # Load and preprocess audio
        audio = load_audio(audio_path, target_sr=16000)
        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        # Process with model
        with torch.no_grad():
            features = self.model.encoder(audio_tensor.unsqueeze(0))
            probabilities = torch.softmax(features, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        # Get predicted class name
        sound_class = self.class_mapping[prediction]

        result = {
            "class": sound_class,
            "confidence": float(confidence),
            "probabilities": {
                self.class_mapping[i]: float(prob)
                for i, prob in enumerate(probabilities[0].cpu().numpy())
            },
        }

        return result


# Example usage
if __name__ == "__main__":
    classifier = SoundClassifier()
    result = classifier.classify("path/to/test/audio.wav")
    print(
        f"Predicted class: {result['class']} with confidence {result['confidence']:.2f}"
    )
