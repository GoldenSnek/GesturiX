import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
from PIL import Image
import io
import uvicorn

# =======================
# 1. CONFIG
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

num_classes = 36
model_path = "model1.pth"

# =======================
# 2. MODEL DEFINITION
# =======================
class SignNet(nn.Module):
    def __init__(self, num_classes=36):
        super(SignNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load model
model = SignNet(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully")

# Class names (0-9, a-z)
class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]

# =======================
# 3. MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# =======================
# 4. HELPER FUNCTIONS
# =======================
def normalize_landmarks(coords):
    """Normalize hand landmarks"""
    wrist = coords[0]
    coords -= wrist
    max_val = np.max(np.abs(coords))
    if max_val > 1e-6:
        coords /= max_val
    return coords

def decode_base64_image(base64_string):
    """Decode base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        return None

def process_frame(frame):
    """Process a single frame and return prediction"""
    try:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(img_rgb)
        
        predictions = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract coordinates
                coords = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks.landmark
                ], dtype=np.float32)
                
                # Normalize
                coords = normalize_landmarks(coords)
                
                # Prepare features for model
                features = torch.tensor(
                    coords.flatten(), 
                    dtype=torch.float32
                ).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(features)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probabilities, 1)
                    
                    label = class_names[pred.item()]
                    conf = confidence.item()
                    
                    predictions.append({
                        'label': label,
                        'confidence': float(conf)
                    })
        
        return predictions
    
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return []

# =======================
# 5. FASTAPI APP
# =======================
app = FastAPI(title="GesturiX Sign Language API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "GesturiX Sign Language Translation API",
        "status": "running",
        "device": str(device)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            
            # Parse JSON
            try:
                frame_data = json.loads(data)
                base64_image = frame_data.get('frame')
                
                if not base64_image:
                    await websocket.send_json({
                        'error': 'No frame data received'
                    })
                    continue
                
                # Decode image
                frame = decode_base64_image(base64_image)
                
                if frame is None:
                    await websocket.send_json({
                        'error': 'Failed to decode image'
                    })
                    continue
                
                # Process frame
                predictions = process_frame(frame)
                
                # Send predictions back
                response = {
                    'predictions': predictions,
                    'detected': len(predictions) > 0
                }
                
                await websocket.send_json(response)
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    'error': 'Invalid JSON format'
                })
                
    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

# =======================
# 6. RUN SERVER
# =======================
if __name__ == "__main__":
    print("🚀 Starting GesturiX Server...")
    print(f"📍 Server will run on http://localhost:8000")
    print(f"🔌 WebSocket endpoint: ws://localhost:8000/ws/translate")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
