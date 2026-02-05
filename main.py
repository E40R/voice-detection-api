from fastapi import FastAPI, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64, tempfile, os
import librosa
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="AI Voice Detection API")

VALID_API_KEY = "sk_test_123456789"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class VoiceRequest(BaseModel):
    language: str = Field(...)
    audioFormat: str = Field(...)
    audioBase64: str = Field(...)

def error(message: str):
    return {"status": "error", "message": message}

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != VALID_API_KEY:
        return False
    return True

def extract_advanced_features(audio: np.ndarray, sr: int) -> Dict:

    features = {}

    try:
        # Reduced MFCC (faster)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features['mfcc_var'] = np.var(mfcc)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]

        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)

        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_std'] = np.std(zcr)

        rms = librosa.feature.rms(y=audio)[0]
        features['rms_var'] = np.var(rms)

        # Pitch analysis (safe)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]

        features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) else 0
        features['pitch_std'] = np.std(pitch_values) if len(pitch_values) else 0

        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_std'] = np.std(spectral_contrast)

    except:
        # safe defaults
        features = {
            'mfcc_var': 0,
            'spectral_bandwidth_mean': 0,
            'zcr_std': 0,
            'rms_var': 0,
            'pitch_mean': 0,
            'pitch_std': 0,
            'spectral_contrast_std': 0
        }

    return features

def calculate_ai_indicators(features: Dict) -> Dict[str, float]:

    pitch_std = features.get('pitch_std', 0)
    pitch_mean = features.get('pitch_mean', 1e-6)
    rms_var = features.get('rms_var', 0)
    bandwidth = features.get('spectral_bandwidth_mean', 0)
    mfcc_var = features.get('mfcc_var', 0)
    zcr_std = features.get('zcr_std', 0)
    contrast_std = features.get('spectral_contrast_std', 0)

    indicators = {}

    pitch_cv = pitch_std / (pitch_mean + 1e-6)
    indicators['pitch_consistency'] = 1 - min(pitch_cv / 0.3, 1)

    indicators['energy_regularity'] = 1 - min(rms_var / 0.01, 1)

    indicators['spectral_flatness'] = 0.7 if bandwidth < 1000 else 0.4 if bandwidth < 2000 else 0.2
    indicators['mfcc_anomaly'] = 0.7 if mfcc_var < 50 else 0.4 if mfcc_var < 100 else 0.2
    indicators['zcr_anomaly'] = 0.7 if zcr_std < 0.02 else 0.4 if zcr_std < 0.05 else 0.2
    indicators['contrast_anomaly'] = 0.7 if contrast_std < 5 else 0.4 if contrast_std < 10 else 0.2

    return indicators

def classify_voice(features: Dict, language: str) -> Tuple[str, float, str]:

    indicators = calculate_ai_indicators(features)

    weights = {
        'pitch_consistency': 0.25,
        'energy_regularity': 0.20,
        'spectral_flatness': 0.15,
        'mfcc_anomaly': 0.15,
        'zcr_anomaly': 0.15,
        'contrast_anomaly': 0.10
    }

    ai_score = sum(indicators[k] * weights[k] for k in weights)

    if ai_score > 0.5:
        classification = "AI_GENERATED"
        confidence = min(ai_score, 0.99)
        explanation = "Detected unnatural pitch consistency and synthetic acoustic patterns"
    else:
        classification = "HUMAN"
        confidence = min(1 - ai_score, 0.99)
        explanation = "Natural human voice variability detected"

    return classification, round(confidence, 2), explanation
@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, api_key: bool = Depends(verify_api_key)):

    if not api_key:
        return error("Invalid API key or malformed request")

    if request.language not in SUPPORTED_LANGUAGES:
        return error("Unsupported language")

    if request.audioFormat.lower() != "mp3":
        return error("Only MP3 format is supported")

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except:
        return error("Invalid base64 encoding")

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio_bytes)
        tmp.close()

        audio, sr = librosa.load(tmp.name, sr=16000, mono=True, duration=25)
        os.remove(tmp.name)

        if len(audio) == 0:
            return error("Empty audio file")

        features = extract_advanced_features(audio, sr)
        classification, confidence, explanation = classify_voice(features, request.language)

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except:
        return error("Audio processing failed")

@app.get("/health")
def health():
    return {"status": "healthy"}
