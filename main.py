from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
import io
import librosa
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated vs Human voices in 5 languages",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VALID_API_KEY = "sk_test_123456789"  # Change this to your secure key
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Pydantic models
class VoiceRequest(BaseModel):
    language: str = Field(..., description="Language of the audio")
    audioFormat: str = Field(..., description="Audio format (mp3)")
    audioBase64: str = Field(..., description="Base64 encoded audio")

class VoiceResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

class ErrorResponse(BaseModel):
    status: str
    message: str

# API Key validation
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key"
        )
    return x_api_key

def extract_advanced_features(audio: np.ndarray, sr: int) -> Dict:
    """
    Extract comprehensive audio features for AI detection
    """
    features = {}
    
    try:
        # 1. MFCC features (captures spectral characteristics)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        features['mfcc_var'] = np.var(mfcc, axis=1)
        
        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # 3. Zero Crossing Rate (voice naturalness)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # 5. RMS Energy (temporal consistency)
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_var'] = np.var(rms)
        
        # 6. Pitch analysis (fundamental frequency)
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
        except:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # 7. Spectral Contrast (AI voices often have flatter contrast)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)
        
        # 8. Tempo and rhythm
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        # Return minimal features on error
        features = {
            'mfcc_mean': np.zeros(40),
            'mfcc_std': np.zeros(40),
            'spectral_centroid_mean': 0,
            'zcr_mean': 0
        }
    
    return features

def calculate_ai_indicators(features: Dict) -> Dict[str, float]:
    """
    Calculate specific indicators that suggest AI generation
    """
    indicators = {}
    
    # 1. Pitch Consistency Score (AI voices have unnaturally consistent pitch)
    if features['pitch_std'] > 0:
        # Low std relative to mean indicates AI
        pitch_cv = features['pitch_std'] / (features['pitch_mean'] + 1e-6)
        indicators['pitch_consistency'] = 1.0 - min(pitch_cv / 0.3, 1.0)  # Normalize
    else:
        indicators['pitch_consistency'] = 0.8  # Very suspicious
    
    # 2. Energy Variance (AI has more regular energy patterns)
    if features['rms_var'] > 0:
        # Low variance indicates AI
        rms_regularity = 1.0 - min(features['rms_var'] / 0.01, 1.0)
        indicators['energy_regularity'] = rms_regularity
    else:
        indicators['energy_regularity'] = 0.9
    
    # 3. Spectral Flatness (AI voices often have flatter spectrum)
    spectral_range = features['spectral_bandwidth_mean']
    if spectral_range < 1000:  # Very narrow bandwidth
        indicators['spectral_flatness'] = 0.7
    elif spectral_range < 2000:
        indicators['spectral_flatness'] = 0.5
    else:
        indicators['spectral_flatness'] = 0.2
    
    # 4. MFCC Pattern Analysis (AI has unusual MFCC distributions)
    mfcc_variance = np.mean(features['mfcc_var'])
    if mfcc_variance < 50:  # Low variance
        indicators['mfcc_anomaly'] = 0.7
    elif mfcc_variance < 100:
        indicators['mfcc_anomaly'] = 0.4
    else:
        indicators['mfcc_anomaly'] = 0.2
    
    # 5. Zero Crossing Rate Consistency
    if features['zcr_std'] < 0.02:  # Very consistent
        indicators['zcr_anomaly'] = 0.7
    elif features['zcr_std'] < 0.05:
        indicators['zcr_anomaly'] = 0.4
    else:
        indicators['zcr_anomaly'] = 0.2
    
    # 6. Spectral Contrast (AI has reduced contrast)
    if features['spectral_contrast_std'] < 5:
        indicators['contrast_anomaly'] = 0.7
    elif features['spectral_contrast_std'] < 10:
        indicators['contrast_anomaly'] = 0.4
    else:
        indicators['contrast_anomaly'] = 0.2
    
    return indicators

def classify_voice(features: Dict, language: str) -> Tuple[str, float, str]:
    """
    Classify voice as AI_GENERATED or HUMAN based on features
    Returns: (classification, confidence_score, explanation)
    """
    # Calculate AI indicators
    indicators = calculate_ai_indicators(features)
    
    # Weighted scoring system
    weights = {
        'pitch_consistency': 0.25,
        'energy_regularity': 0.20,
        'spectral_flatness': 0.15,
        'mfcc_anomaly': 0.15,
        'zcr_anomaly': 0.15,
        'contrast_anomaly': 0.10
    }
    
    # Calculate AI probability
    ai_score = sum(indicators[key] * weights[key] for key in weights.keys())
    
    # Determine classification
    threshold = 0.5
    if ai_score > threshold:
        classification = "AI_GENERATED"
        confidence = min(ai_score, 0.99)  # Cap at 0.99
        
        # Generate explanation based on strongest indicators
        explanations = []
        if indicators['pitch_consistency'] > 0.6:
            explanations.append("unnatural pitch consistency")
        if indicators['energy_regularity'] > 0.6:
            explanations.append("robotic speech patterns")
        if indicators['spectral_flatness'] > 0.6:
            explanations.append("synthetic spectral characteristics")
        if indicators['mfcc_anomaly'] > 0.6:
            explanations.append("artificial acoustic signatures")
        
        if not explanations:
            explanations = ["algorithmic voice generation patterns detected"]
        
        explanation = f"Detected {', '.join(explanations[:2])}"
        
    else:
        classification = "HUMAN"
        confidence = min(1.0 - ai_score, 0.99)
        
        # Generate explanation for human classification
        explanations = []
        if indicators['pitch_consistency'] < 0.4:
            explanations.append("natural pitch variations")
        if indicators['energy_regularity'] < 0.4:
            explanations.append("organic speech dynamics")
        if indicators['spectral_flatness'] < 0.4:
            explanations.append("human vocal characteristics")
        
        if not explanations:
            explanations = ["natural human voice patterns"]
        
        explanation = f"Identified {', '.join(explanations[:2])}"
    
    return classification, round(confidence, 2), explanation

@app.post("/api/voice-detection", response_model=VoiceResponse)
async def detect_voice(
    request: VoiceRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main endpoint for voice detection
    """
    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Must be one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    
    # Validate audio format
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 format is supported"
        )
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audioBase64)
        
        # Load audio using librosa
        audio_io = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_io, sr=16000, mono=True)
        
        # Check if audio is valid
        if len(audio) == 0:
            raise ValueError("Empty audio file")
        
        # Extract features
        features = extract_advanced_features(audio, sr)
        
        # Classify
        classification, confidence, explanation = classify_voice(features, request.language)
        
        return VoiceResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation
        )
    
    except base64.binascii.Error:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 encoding"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "AI Voice Detection API",
        "supported_languages": SUPPORTED_LANGUAGES
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "supported_languages": SUPPORTED_LANGUAGES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)