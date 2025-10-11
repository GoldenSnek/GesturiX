import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import base64
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# =======================
# 1. APP CONFIG
# =======================
app = FastAPI()

# Allow all origins for development purposes (you might want to restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# 2. MODEL CONFIG & LOADING
# =======================
device = torch.device("cpu") # Use CPU for broader compatibility on servers
print("✅ Using device:", device)

num_classes = 36
model_path = "model1.pth" # Ensure this path is correct relative to where you run the server

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

model = SignNet(num_classes).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}. Please check the path.")
    exit()


class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]

# =======================
# 3. MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, # Process one hand for clarity
    min_detection_confidence=0.7
)

# =======================
# 4. HELPER & PROCESSING FUNCTIONS
# =======================
def normalize_landmarks(coords):
    wrist = coords[0]
    coords -= wrist
    # Using L2 norm for scaling instead of max absolute value for more robust normalization
    norm_val = np.linalg.norm(coords)
    if norm_val > 0:
        coords /= norm_val
    return coords

def process_image(image_bytes: bytes) -> str:
    """
    Decodes image bytes, processes landmarks, and returns the predicted sign.
    """
    try:
        # Decode the image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return "Error decoding image"

        # The model expects a certain orientation. Let's flip it to match a selfie camera.
        frame = cv2.flip(frame, 1)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert to coordinates
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                
                # Normalize and prepare for model
                coords_normalized = normalize_landmarks(coords.copy())
                features = torch.tensor(coords_normalized.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

                # Model prediction
                with torch.no_grad():
                    outputs = model(features)
                    _, pred = torch.max(outputs, 1)
                    label = class_names[pred.item()]
                    return label # Return the first prediction found
        
        return "" # Return empty string if no hand is detected
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""


# =======================
# 5. WEBSOCKET ENDPOINT
# =======================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connection established.")
    try:
        while True:
            # Receive base64 image data from the client
            base64_str = await websocket.receive_text()
            print(f"-> Received frame of size: {len(base64_str)} bytes")
            
            # Decode the base64 string to bytes
            image_bytes = base64.b64decode(base64_str)
            
            # Process the image and get the prediction
            label = process_image(image_bytes)
            
            # Send the result back to the client
            if label:
                await websocket.send_text(label)

    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
    finally:
        print("WebSocket connection closed.")


# To run the server, use the command: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
