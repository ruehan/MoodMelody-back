from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from fer import FER
import cv2
import tensorflow as tf
from typing import List

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
            return {"emotion": "No face detected"}

        # 가장 강한 감정 추출
        dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
        emotion, score = dominant_emotion

        return {"emotion": emotion, "confidence": float(score)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/recommend_music")
async def recommend_music(emotion: str):
    # 실제 구현에서는 데이터베이스나 외부 API를 사용하여 음악을 추천해야 함
    music_recommendations = {
        "angry": {"title": "Thunderstruck", "artist": "AC/DC"},
        "disgust": {"title": "Creep", "artist": "Radiohead"},
        "fear": {"title": "Thriller", "artist": "Michael Jackson"},
        "happy": {"title": "Don't Stop Me Now", "artist": "Queen"},
        "sad": {"title": "Someone Like You", "artist": "Adele"},
        "surprise": {"title": "Wow", "artist": "Post Malone"},
        "neutral": {"title": "Comfortably Numb", "artist": "Pink Floyd"}
    }
    
    recommended_music = music_recommendations.get(emotion, {"title": "Unknown", "artist": "Unknown"})
    return recommended_music

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)