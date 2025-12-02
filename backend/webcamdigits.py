import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import random # Needed for PredictionSmoother

# =======================
# 1. CONFIG
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

num_classes = 10 
# Use the final saved model name
model_path = "backend/model_digits_optimized.pth" 
# Use the best model saved by the Early Stopping mechanism if available
best_model_path = "backend/best_model_digits_optimized.pth" 

confidence_threshold = 0.75 # Increased confidence threshold for high accuracy model

# =======================
# 2. MODEL DEFINITION (MATCHING THE DEEPER ARCHITECTURE)
# =======================
class SignNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SignNet, self).__init__()
        # EXACT ARCHITECTURE from the training script (5 layers deep)
        self.fc = nn.Sequential(
            nn.Linear(63, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
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
            nn.Linear(128, num_classes) # Output: 10 classes
        )

    def forward(self, x):
        return self.fc(x)

# Load model
print("📂 Loading model...")
model = SignNet(num_classes).to(device)

# Prioritize loading the 'best' model found during Early Stopping
try:
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✅ Loaded BEST model: {best_model_path}")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded final model: {model_path}")
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Model files not found at {best_model_path} or {model_path}.")
        print("Please run the training script first or update the model_path.")
        exit()

model.eval()

# Class names (Digits 0-9)
class_names = [str(i) for i in range(10)]

# =======================
# 3. MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, # Optimized for two hands if needed, but only uses the first one
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# =======================
# 4. HELPER FUNCTIONS (MATCHING THE IMPROVED NORMALIZATION)
# =======================
def normalize_landmarks(coords):
    """Normalize hand landmarks relative to wrist, with independent Z-axis scaling."""
    # 1. Center coordinates relative to the wrist (landmark 0)
    wrist = coords[0]
    coords_centered = coords - wrist
    
    # 2. Global XY Scaling (using max absolute value of XY)
    xy_coords = coords_centered[:, :2]
    scale_factor_xy = np.max(np.abs(xy_coords)) + 1e-6
    coords_centered[:, :2] /= scale_factor_xy

    # 3. Z-axis Normalization 
    z_coords = coords_centered[:, 2]
    z_range = np.max(z_coords) - np.min(z_coords) + 1e-6
    coords_centered[:, 2] = (z_coords - np.mean(z_coords)) / (z_range / 2)
    
    return coords_centered

class PredictionSmoother:
    """Smooth predictions to reduce jitter using a majority vote window."""
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.predictions = []
    
    def update(self, pred_idx):
        self.predictions.append(pred_idx)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        
        if self.predictions:
            return max(set(self.predictions), key=self.predictions.count)
        return pred_idx

smoother = PredictionSmoother(window_size=7)

# =======================
# 5. WEBCAM LOOP
# =======================
print("🎥 Starting webcam. Press 'q' to quit.")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Top banner text
    cv2.putText(frame, "GesturiX: Optimized ASL Digits (0-9)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Model: {model_path} | Threshold: {confidence_threshold*100:.0f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Feature extraction (using the improved normalization function)
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            coords = normalize_landmarks(coords)
            features = torch.tensor(coords.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

            # Model prediction
            with torch.no_grad():
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probabilities, 1)
                
                pred_idx = pred.item()
                conf_item = confidence.item()

            # Smoothing and Confidence Check
            if conf_item > confidence_threshold:
                smoothed_idx = smoother.update(pred_idx)
                label = class_names[smoothed_idx]
                conf_percent = conf_item * 100
                color = (0, 255, 0) # Green
            else:
                label = "?"
                conf_percent = conf_item * 100
                color = (0, 165, 255) # Orange
                smoother.update(pred_idx) 

            # Draw Landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Draw Label
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h) - 40
            
            text = f"DIGIT: {label} ({conf_percent:.0f}%)"
            
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, 
                          (x - 5, y - text_height - 10), 
                          (x + text_width + 5, y + 5),
                          (0, 0, 0), -1)
                          
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)
    else:
         cv2.putText(frame, "No hands detected", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


    cv2.imshow("GesturiX Live Sign Translation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")