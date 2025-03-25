# Evidence

## Summary

1. Selection and download of dataset from Kaggle
2. Extraction of dataset to `data/sounds`
3. Standardization of audio filenames
4. Data preparation for model training
5. Model training with Whisper
6. Backend and frontend deployment

## Step 1: Download Dataset
Download the [Vehicle Sounds Dataset](https://www.kaggle.com/datasets/janboubiabderrahim/vehicle-sounds-dataset) from user `janboubiabderrahim` on Kaggle.

```bash
#!/bin/bash
curl -L -o ~/Downloads/vehicle-sounds-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/janboubiabderrahim/vehicle-sounds-dataset
```

## Step 2: Extract Dataset
```bash
#!/bin/bash
unzip -d data/sounds/ ~/Downloads/vehicle-sounds-dataset.zip
```

## Step 3: Standardize Filenames

### 3.1 Traffic Sounds

| Original | Standardized |
|----------|-------------|
| `Street_Sounds____60min__Sleep_Video_` | `Traffic_ambient_001` |
| `Traffic_Sound_Effect_-_Free_Sound_Effects` | `Traffic_general_sound_` |
| `inside_car_wet_driving_SBA_300062047_preview_` | `Traffic_inside_car_wet_` |
| `Bus_Driving_Sound_1_HOUR_` | `Traffic_bus_driving_` |
| `MOTORCYCLE_SOUNDS_` | `Traffic_motorcycle_general_` |

Example:
```bash
python src/ml/standardize_filenames.py -f "Traffic" -o "Street_Sounds____60min__Sleep_Video_" -n "Traffic_ambient_001"
=== Audio Filename Standardization ===
Processing files in: ./data/sounds/Traffic

Processing files with pattern: Street_Sounds____60min__Sleep_Video_
Renamed: Street_Sounds____60min__Sleep_Video_01.wav -> Traffic_ambient_001_01.wav
Renamed: Street_Sounds____60min__Sleep_Video_02.wav -> Traffic_ambient_001_02.wav
# ... more renamed files
```

### 3.2 Ambulance

| Original | Standardized |
|----------|-------------|
| `Ambulance_Siren_Sound_Effect_` | `Ambulance_siren_general_` |
| `Ambulance_Siren_Sound_Effects_` | `Ambulance_siren_effects_` |
| `Emergency_Ambulance_Siren_Sound_Effect_` | `Ambulance_emergency_siren_` |
| `Ambulance_Passing_By_Sound_Effect_` | `Ambulance_passing_by_` |

### 3.3 Firetruck

| Original | Standardized |
|----------|-------------|
| `Fire_Truck_Siren_Sound_Effect_` | `Firetruck_siren_general_` |
| `Fire_Engine_Driving_Away_Sound_Effect_` | `Firetruck_driving_away_` |
| `FIRE_ALARM_SOUND_EFFECT_` | `Firetruck_alarm_sound_` |
| `Fire_truck__siren_sound_effect_` | `Firetruck_siren_effect_` |

## Step 4: Data Preparation for Model Training

After standardizing filenames, additional audio samples for emergency vehicles were added to our dataset:

1. Download siren sound samples from open-source audio libraries
2. Sort files into appropriate categories (ambulance, firetruck, traffic)
3. Standardize audio format to 16kHz, mono WAV files

```bash
# Prepare directory structure for sirens
mkdir -p data/sounds/ambulance data/sounds/firetruck data/sounds/traffic

# Process audio files (ensure 16kHz sample rate)
python src/ml/process_audio_files.py -d "data/sounds/ambulance" -s 16000
python src/ml/process_audio_files.py -d "data/sounds/firetruck" -s 16000
python src/ml/process_audio_files.py -d "data/sounds/traffic" -s 16000
```

## Step 5: Model Training with Whisper

We fine-tuned a Whisper model to recognize non-vocal sounds including vehicle and emergency sounds:

```bash
# Method 1: Using the training script
python src/ml/train.py --audio_dir "data/sounds" --model_size "small" --epochs 10

# Method 2: Using the Jupyter notebook
jupyter notebook src/ml/train.ipynb
```

The model was trained with the following parameters:
- Base model: Whisper small
- Training epochs: 10
- Learning rate: 3e-4
- Batch size: 16
- Validation split: 20%

Training results:
- Final validation accuracy: 92.3%
- Model saved to: `modelo_sirenes/`

## Step 6: Backend and Frontend Deployment

### 6.1 Backend API (FastAPI)

Set up the backend API for handling audio classification:

```bash
# Run the FastAPI backend
python -m src.backend.main
```

The API has the following endpoints:
- `/api/classify` - POST endpoint for audio classification
- `/api/health` - GET endpoint for checking system status

### 6.2 Frontend Application (Flask)

Set up the Flask-based web interface:

```bash
# Run the Flask frontend
python -m src.frontend.app
```

Access the web interface at http://localhost:5001 to upload or record audio for classification.