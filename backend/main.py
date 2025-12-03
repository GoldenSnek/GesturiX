from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
import uvicorn

# ================= CONFIG =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 29  # model2 (A-Z + del + nothing + space)
model_path = "model2.pth"

# ================= MODEL =================
class SignNet(nn.Module):
    def __init__(self, num_classes=29):
        super(SignNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(63, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

model = SignNet(num_classes).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded {model_path} successfully")
except FileNotFoundError:
    print(f"❌ Error: {model_path} not found. Make sure the file is in the same directory.")

class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ['del', 'nothing', 'space']

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8 # change para balance speed vs accuracy
)

def normalize_landmarks(coords):
    wrist = coords[0]
    coords -= wrist
    coords /= np.max(np.abs(coords) + 1e-6)
    return coords

# ================= GEMINI =================
load_dotenv() 

try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
         print("Warning: GEMINI_API_KEY not found in environment.")
    else:
        client = genai.Client(api_key=api_key)
        print("Gemini Client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Gemini Client: {e}")

class EnhanceRequest(BaseModel):
    raw_text: str

# ================= FASTAPI =================
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        
        label = class_names[pred.item()]

    return {"prediction": label}

@app.post("/enhance")
async def enhance_translation(request: EnhanceRequest):
    raw_text = request.raw_text.strip()
    if not raw_text:
        return {"enhanced_text": "Please sign a sentence first."}

    prompt = f"""
    You are an AI Sign Language Interpreter Assistant. Your goal is to refine raw, literal translation output from a sign language recognition system into fluent, natural, and grammatically correct English.

    ### CORE TASK:
    1.  **Correct Grammar and Syntax**: Introduce proper punctuation (commas, periods, question marks) and capitalization.
    2.  **Ensure Fluency**: Interpolate implied words (like "am," "is," "the," "a") that are often omitted in signed or raw, letter-by-letter translation.
    3.  **Maintain Meaning**: Preserve the original core intent of the signed phrase.

    ### CONSTRAINTS (Strictly Follow These):
    * **DO NOT** add any extra commentary, explanations, or labels (e.g., "Enhanced Sentence:").
    * **OUTPUT ONLY** the single, refined sentence.
    * The final output must be ready to be spoken or displayed immediately.

    ### RAW INPUT TO REFINE:
    {raw_text}
    """

    try:
        if 'client' not in globals():
            return {"enhanced_text": "AI Enhancement unavailable (API Key missing)."}

        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt
        )
        enhanced_text = response.text.strip()
        return {"enhanced_text": enhanced_text}
    except APIError as e:
        print(f"Gemini API Error: {e}")
        return {"enhanced_text": "AI Enhancement failed. API Error."}
    except Exception as e:
        print(f"General Error during enhancement: {e}")
        return {"enhanced_text": "AI Enhancement failed. Please try again."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)