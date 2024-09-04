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

# 감정 분석 모델 로드 (예시)
# 실제로는 훈련된 모델을 로드해야 합니다
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

# 감정 레이블
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

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
    try:
        # 바이트 데이터를 numpy 배열로 변환
        nparr = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 이미지 전처리
        img = cv2.resize(img, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (1, 48, 48, 1))
        
        # 감정 예측 (실제 모델로 대체 필요)
        # prediction = model.predict(img)
        # emotion_index = np.argmax(prediction)
        # emotion = emotion_labels[emotion_index]
        
        # 테스트를 위해 임시로 랜덤한 감정 반환
        emotion = np.random.choice(emotion_labels)
        
        return JSONResponse(content={"emotion": emotion}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/recommend_music")
async def recommend_music(emotion: str):
    # 실제 구현에서는 데이터베이스나 외부 API를 사용하여 음악을 추천해야 합니다
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