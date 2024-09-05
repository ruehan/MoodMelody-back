import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from fer import FER
import cv2
import tensorflow as tf
from typing import List
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
emotion_detector = FER(mtcnn=True)

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

emotion_to_features = {
    'angry': {'valence': (0.0, 0.4), 'energy': (0.8, 1.0)},
    'disgust': {'valence': (0.0, 0.3), 'energy': (0.4, 0.7)},
    'fear': {'valence': (0.0, 0.3), 'energy': (0.5, 0.8)},
    'happy': {'valence': (0.7, 1.0), 'energy': (0.7, 1.0)},
    'sad': {'valence': (0.0, 0.3), 'energy': (0.0, 0.3)},
    'surprise': {'valence': (0.5, 1.0), 'energy': (0.7, 1.0)},
    'neutral': {'valence': (0.4, 0.6), 'energy': (0.4, 0.6)}
}

def recommend_music(emotion: str) -> List[Dict]:
    features = emotion_to_features.get(emotion.lower())
    if not features:
        raise HTTPException(status_code=400, detail="Invalid emotion")

    recommendations = sp.recommendations(
        seed_genres=['pop', 'rock'],
        limit=5,
        target_valence=(features['valence'][0] + features['valence'][1]) / 2,
        target_energy=(features['energy'][0] + features['energy'][1]) / 2
    )

    tracks = []
    for track in recommendations['tracks']:
        tracks.append({
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'preview_url': track['preview_url'],
            'external_url': track['external_urls']['spotify']
        })

    return tracks

@app.post("/analyze_emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 감정 분석 수행
        emotions = emotion_detector.detect_emotions(img)

        if not emotions:
            return {"emotion": "No face detected", "confidence": 0, "tracks": []}

        # 가장 강한 감정 추출
        dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
        emotion, confidence = dominant_emotion

        # 음악 추천
        tracks = recommend_music(emotion)

        return {
            "emotion": emotion,
            "confidence": float(confidence),
            "tracks": tracks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)