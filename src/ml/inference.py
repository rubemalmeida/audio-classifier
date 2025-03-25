import torch
import numpy as np
import librosa
import os
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration, logging
from torch.nn.functional import softmax

DEFAULT_MODEL_PATH = "openai/whisper-small"

class SoundClassifier:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,  # Changed to use the pretrained model directly
    ):
        # Check if model_path is a directory and doesn't have the required files
        if os.path.isdir(model_path) and not os.path.exists(
            os.path.join(model_path, "preprocessor_config.json")
        ):
            print(
                f"Warning: {model_path} does not contain required model files, using default Whisper model"
            )
            model_path = DEFAULT_MODEL_PATH

        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

    def classify(self, audio_path):
        """Classify the sound in the given audio file"""
        audio, sr = librosa.load(audio_path, sr=16000)
        target_length = 16000 * 30
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():                
            outputs = self.model.generate(
                inputs.input_features,
                language="en",
                task="transcribe",
                use_cache=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        token_ids = outputs.sequences[0]
        scores = outputs.scores

        transcription = self.processor.decode(token_ids)
        transcription = transcription.replace("<|startoftranscript|>", "") \
                                    .replace("<|en|>", "") \
                                    .replace("<|transcribe|>", "") \
                                    .replace("<|notimestamps|>", "") \
                                    .replace("<|endoftext|>", "")\
                                    .strip()

        token_probabilities = []
        for i, score_tensor in enumerate(scores):
            probabilities = torch.softmax(score_tensor, dim=-1)
            top_probs, top_indices = torch.topk(probabilities[0], k=2) #get the top 5
            top_tokens = [self.processor.decode(idx.item()) for idx in top_indices]
            top_probabilities = [prob * 100 for prob in top_probs.cpu().numpy().tolist()]
            
            if top_tokens[0] == "<|endoftext|>":
                break

            # token_probabilities.append(top_tokens[0])
            token_probabilities.append(top_probabilities[0])
            
        #return transcription, token_probabilities
        confidence = sum(token_probabilities) / len(token_probabilities)
        result = {
            "class": transcription,
            "confidence": confidence,
        }
        return result

