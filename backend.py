import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from deepface import DeepFace
import cv2
from typing import List, Dict
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import random

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

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def analyze_emotion_deepface(img):
    try:
        result = DeepFace.analyze(img, actions=['emotion'])
        emotions = result[0]['emotion']
        dominant_emotion = max(emotions, key=emotions.get)
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepFace analysis failed: {str(e)}")

def emotion_to_music_features(emotion_scores):
    # 감정 점수 정규화
    total_score = sum(emotion_scores.values())
    normalized_scores = {k: v / total_score for k, v in emotion_scores.items()}
    
    # 각 감정이 음악 특성에 미치는 영향 정의
    emotion_impact = {
        'angry': {'valence': -0.8, 'energy': 0.9, 'danceability': 0.4, 'tempo': 0.8},
        'disgust': {'valence': -0.5, 'energy': 0.4, 'danceability': -0.3, 'tempo': 0.2},
        'fear': {'valence': -0.7, 'energy': 0.6, 'danceability': -0.2, 'tempo': 0.7},
        'happy': {'valence': 0.9, 'energy': 0.7, 'danceability': 0.8, 'tempo': 0.5},
        'sad': {'valence': -0.7, 'energy': -0.5, 'danceability': -0.4, 'tempo': -0.3},
        'surprise': {'valence': 0.4, 'energy': 0.6, 'danceability': 0.5, 'tempo': 0.6},
        'neutral': {'valence': 0, 'energy': 0, 'danceability': 0, 'tempo': 0}
    }
    
    # 각 특성에 대한 가중 평균 계산
    features = {}
    for feature in ['valence', 'energy', 'danceability', 'tempo']:
        feature_score = sum(normalized_scores[emotion] * emotion_impact[emotion][feature] 
                            for emotion in emotion_scores)
        # 점수를 0-1 범위로 조정
        features[feature] = (feature_score + 1) / 2
    
    # tempo를 BPM 범위로 변환
    min_bpm, max_bpm = 60, 180
    tempo_range = max_bpm - min_bpm
    tempo_bpm = min_bpm + (features['tempo'] * tempo_range)
    
    return {
        'target_valence': max(0, min(features['valence'], 1)),
        'target_energy': max(0, min(features['energy'], 1)),
        'target_danceability': max(0, min(features['danceability'], 1)),
        'target_tempo': tempo_bpm,
        'min_tempo': max(min_bpm, tempo_bpm - 20),
        'max_tempo': min(max_bpm, tempo_bpm + 20)
    }

def recommend_music(emotion_scores):
    features = emotion_to_music_features(emotion_scores)
    
    # 수정된 공통 장르 풀 정의
    common_genres = ['pop', 'rock', 'indie', 'hip-hop', 'r-n-b', 'jazz', 'k-pop', 'folk']
    
    # 감정에 따라 장르 가중치 조정
    genre_weights = adjust_genre_weights(emotion_scores, common_genres)
    
    # 가중치에 따라 장르 선택 (5개)
    selected_genres = random.choices(common_genres, weights=genre_weights, k=5)
    
    recommendations = sp.recommendations(
        seed_genres=selected_genres,
        limit=50,  # 더 많은 트랙을 요청하여 필터링의 여지를 둠
        **features
    )

    tracks = []
    for track in recommendations['tracks']:
        if track['preview_url']:  # 미리 듣기 URL이 있는 경우에만 처리
            audio_features = sp.audio_features(track['id'])[0]
            if audio_features:
                match_score = calculate_match_score(features, audio_features)
                tracks.append({
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'preview_url': track['preview_url'],
                    'external_url': track['external_urls']['spotify'],
                    'match_score': match_score
                })

    # 매치 점수에 따라 정렬하고 상위 5개 선택
    tracks.sort(key=lambda x: x['match_score'], reverse=True)
    return tracks[:5]

def adjust_genre_weights(emotion_scores, genres):
    # 감정 점수 정규화
    total_score = sum(emotion_scores.values())
    normalized_scores = {k: v / total_score for k, v in emotion_scores.items()}
    
    # 장르별 기본 가중치 (수정된 장르 목록에 맞춰 조정)
    genre_base_weights = {
        'pop': 1, 'rock': 1, 'indie': 1, 'hip-hop': 1,
        'r-n-b': 1, 'jazz': 1, 'k-pop': 1, 'folk': 1
    }
    
    # 감정에 따른 장르 가중치 조정 (k-pop 추가 및 조정)
    emotion_genre_impact = {
        'angry': {'rock': 0.3, 'hip-hop': 0.2, 'indie': 0.1},
        'disgust': {'rock': 0.2, 'indie': 0.2, 'jazz': 0.1},
        'fear': {'jazz': 0.3, 'folk': 0.2, 'indie': 0.1},
        'happy': {'pop': 0.3, 'k-pop': 0.2, 'r-n-b': 0.1},
        'sad': {'indie': 0.3, 'folk': 0.2, 'jazz': 0.1},
        'surprise': {'k-pop': 0.3, 'pop': 0.2, 'hip-hop': 0.1},
        'neutral': {'pop': 0.2, 'rock': 0.2, 'k-pop': 0.1}
    }
    
    # 감정에 따라 장르 가중치 조정
    for emotion, score in normalized_scores.items():
        for genre, impact in emotion_genre_impact.get(emotion, {}).items():
            if genre in genre_base_weights:
                genre_base_weights[genre] += score * impact
    
    # 최종 가중치 계산 및 정규화
    total_weight = sum(genre_base_weights.values())
    normalized_weights = [genre_base_weights[genre] / total_weight for genre in genres]
    
    return normalized_weights

def calculate_match_score(target_features, track_features):
    score = 0
    feature_weights = {'valence': 0.3, 'energy': 0.3, 'danceability': 0.2, 'tempo': 0.2}
    
    for feature, weight in feature_weights.items():
        if feature == 'tempo':
            if target_features['min_tempo'] <= track_features[feature] <= target_features['max_tempo']:
                tempo_score = 1 - abs(track_features[feature] - target_features['target_tempo']) / (target_features['max_tempo'] - target_features['min_tempo'])
            else:
                tempo_score = 0
            score += weight * tempo_score
        else:
            score += weight * (1 - abs(target_features[f'target_{feature}'] - track_features[feature]))

    return score  # 이미 0-1 사이의 점수

@app.post("/analyze_emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # DeepFace를 사용한 감정 분석
        emotion_result = analyze_emotion_deepface(img)

        # 음악 추천
        tracks = recommend_music(emotion_result['emotion_scores'])

        return {
            **emotion_result,
            "tracks": tracks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)