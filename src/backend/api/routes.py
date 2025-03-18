import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
from src.backend.audio_processing import process_audio
from src.ml.inference import SoundClassifier

router = APIRouter()
classifier = SoundClassifier('./src/ml/modelo_sirenes')


@router.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    """Classify the uploaded audio file"""
    # Validate file type
    if not file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .wav or .mp3 file.",
        )

    # Save uploaded file to temp location
    temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process audio (ensure 16kHz, max 30 seconds)
        processed_file = process_audio(temp_file)

        # Classify the sound
        result = classifier.classify(processed_file)

        # Clean up temporary files
        if temp_file != processed_file:
            os.remove(temp_file)
        os.remove(processed_file)

        return JSONResponse(content=result)

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@router.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "model_loaded": True}
