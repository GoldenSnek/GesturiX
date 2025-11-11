# main.py
from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp

# ================= CONFIG =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 36
model_path = "model1.pth"

# ================= MODEL =================
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
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ================= FASTAPI =================
app = FastAPI()

def normalize_landmarks(coords):
    wrist = coords[0]
    coords -= wrist
    coords /= np.max(np.abs(coords) + 1e-6)
    return coords

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return {"prediction": "None"}

    landmarks = results.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    coords = normalize_landmarks(coords)
    features = torch.tensor(coords.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(features)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    return {"prediction": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)