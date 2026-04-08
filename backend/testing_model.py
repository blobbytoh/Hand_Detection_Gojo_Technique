import pickle
import numpy as np
import string
import pandas as pd
import warnings
from backend.main import open_palm, distance, extract_normalized_features

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json

warnings.filterwarnings("ignore", category=UserWarning)

class NoCacheStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

app = FastAPI()
templates = Jinja2Templates(directory='frontend')
app.mount("/static", NoCacheStaticFiles(directory='frontend/static'), name='static')

model_single = pickle.load(open('backend/gesture_model_single.pkl', 'rb'))
model_dual = pickle.load(open('backend/gesture_model_dual.pkl', 'rb'))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            landmarks = json.loads(data)
            
            if landmarks.get("type") != "landmarks":
                continue
            
            landmarks_list = landmarks.get("data", [])
            
            hands_data = []
            raw_landmarks = []
            
            for hand_landmarks in landmarks_list:
                class LandMark:
                    def __init__(self, d):
                        self.x = d['x']
                        self.y = d['y']
                        
                lm_objects = [LandMark(lm) for lm in hand_landmarks]
                
                features, max_dist = extract_normalized_features(lm_objects)
                
                hands_data.append({
                    "features": features,
                    "max_dist": max_dist,
                    "open_palm": open_palm(lm_objects)
                })
                
                raw_landmarks.append(lm_objects)
                
            result = {
                "sign": "No Hand Detected",
                'confident': 0.0,
                'hand_count': len(hands_data)
            }
            
            if len(hands_data) == 2:
            # Dual-hand: Check for Hollow Purple Prepare
                left_features = hands_data[0]['features']
                right_features = hands_data[1]['features']
                
                # Calculate wrist distance
                left_wrist = raw_landmarks[0][0]
                right_wrist = raw_landmarks[1][0]
                wrist_dist = distance(left_wrist, right_wrist)
                
                # Normalize by average hand scale
                scale = (hands_data[0]['max_dist'] + hands_data[1]['max_dist']) / 2
                if scale == 0:
                    scale = 1
                
                # Build 85-dim feature vector
                fused_features = left_features + right_features + [wrist_dist / scale]
                
                # Verify
                if len(fused_features) != 85:
                    prediction_text = f"Error: {len(fused_features)} features"
                else:
                    # One-Class SVM prediction
                    result = model_dual.predict([fused_features])[0]  # +1 or -1
                    score = model_dual.decision_function([fused_features])[0]
                    
                    if result == 1 and score > 0:
                        result = {
                            "sign": "Hollow Purple Prepare", 
                            "confidence": min(abs(score) * 50, 100) / 100,
                            "hand_count": 2
                        }
                    else:
                        result = {
                            "sign": "Not Merge",
                            "confidence": 0,
                            "hand_count": 2
                        }

            elif len(hands_data) == 1:
                # Single-hand: Use standard classifier
                features = hands_data[0]['features']
                
                if len(features) != 42:
                    prediction_text = f"Error: {len(features)} features"
                else:
                    input_df = pd.DataFrame([features])
                    probabilities = model_single.predict_proba(input_df)[0]
                    max_prob = np.max(probabilities)
                    predicted_class = model_single.classes_[np.argmax(probabilities)]
                    
                    if max_prob > 0.1:
                        result = {
                            "sign": predicted_class,
                            "confidence": float(max_prob),
                            "hand_count": 1
                        }
                    else:
                        result = {
                            "sign": "Uncertain",
                            "confidence": float(max_prob),
                            "hand_count": 1
                        }
                
            await websocket.send_text(json.dumps(result))
                
    except WebSocketDisconnect:
        print('Web socket disconnected')
    
if __name__ == "__main__":
    uvicorn.run('testing_model:app', host='127.0.0.1', port=8080, reload=True)