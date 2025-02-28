import os
import tempfile
import numpy as np
import soundfile as sf
from pydub import AudioSegment


def ensure_16khz(file_path, output_path=None):
    """
    Ensure audio is at 16kHz sample rate
    Returns the path to the processed file
    """
    # If no output path specified, create temporary file
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "processed_audio.wav")

    # Get audio data and sample rate
    data, sample_rate = sf.read(file_path)

    # If sample rate is already 16kHz, just return the file path
    if sample_rate == 16000:
        if file_path != output_path:
            sf.write(output_path, data, 16000)
        return output_path

    # Otherwise, convert to 16kHz
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


def trim_audio(file_path, max_duration=30, output_path=None):
    """
    Trim audio to max_duration seconds (if longer)
    Returns the path to the processed file
    """
    # If no output path specified, create temporary file
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "trimmed_audio.wav")

    # Load audio and check duration
    audio = AudioSegment.from_file(file_path)

    # If shorter than max duration, just return the file path
    if len(audio) <= max_duration * 1000:  # pydub uses milliseconds
        if file_path != output_path:
            audio.export(output_path, format="wav")
        return output_path

    # Otherwise, trim audio
    trimmed_audio = audio[: max_duration * 1000]
    trimmed_audio.export(output_path, format="wav")

    return output_path


def process_audio(file_path, output_path=None):
    """
    Process audio to ensure it's 16kHz and max 30 seconds
    Returns the path to the processed file
    """
    # If no output path specified, create one
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "processed_audio.wav")

    # First trim the audio
    trimmed_path = trim_audio(file_path, max_duration=30)

    # Then ensure it's at 16kHz
    processed_path = ensure_16khz(trimmed_path, output_path)

    # Clean up temporary file if needed
    if trimmed_path != file_path and trimmed_path != processed_path:
        os.remove(trimmed_path)

    return processed_path
