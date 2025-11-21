# main.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import os
from dotenv import load_dotenv # <-- NEW IMPORT
from google import genai
from google.genai import types
from google.genai.errors import APIError
import uvicorn

# ================= CONFIG (SAME AS BEFORE) =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 36
model_path = "model1.pth"

# ================= MODEL (SAME AS BEFORE) =================
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

# ================= MEDIAPIPE (SAME AS BEFORE) =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def normalize_landmarks(coords):
    wrist = coords[0]
    coords -= wrist
    coords /= np.max(np.abs(coords) + 1e-6)
    return coords

# ================= GEMINI SETUP (NEW) =================
load_dotenv() 

try:
    # 2. Get the key from the environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
         raise ValueError("GEMINI_API_KEY not found in environment or .env file.")

    # 3. Initialize client using the key
    client = genai.Client(api_key=api_key)
    print("Gemini Client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Gemini Client: {e}")
    # You might want to handle this error more gracefully in a production environment

class EnhanceRequest(BaseModel):
    raw_text: str

# ================= FASTAPI =================
app = FastAPI()

# Your existing /predict endpoint (no change needed)
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

# NEW: /enhance endpoint to use Gemini
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

    ### EXAMPLE:
    Raw Translation: "I H U N G R Y NOW W A N T FOOD"
    Refined Output: "I am hungry now, I want food."

    ### RAW INPUT TO REFINE:
    {raw_text}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Fast and great for text editing tasks
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