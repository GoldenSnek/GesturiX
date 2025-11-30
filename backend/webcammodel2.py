import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp

# =======================
# 1. CONFIG
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

num_classes = 29  # Updated to 29 classes (A-Z + del + nothing + space)
model_path = "backend/model2.pth"# Updated path

# =======================
# 2. MODEL DEFINITION (Updated Architecture)
# =======================
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

# Load model
print("📂 Loading model...")
model = SignNet(num_classes).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Model not found at: {model_path}")
    print("Please update the model_path variable with the correct path.")
    exit()

# Class names (A-Z + del + nothing + space)
class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ['del', 'nothing', 'space']

# =======================
# 3. MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Changed to 2 for better performance
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# =======================
# 4. HELPER FUNCTIONS
# =======================
def normalize_landmarks(coords):
    """Normalize hand landmarks relative to wrist"""
    wrist = coords[0]
    coords -= wrist
    coords /= np.max(np.abs(coords) + 1e-6)
    return coords

# =======================
# 5. PREDICTION SMOOTHING
# =======================
class PredictionSmoother:
    """Smooth predictions to reduce jitter"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = []
    
    def update(self, pred):
        self.predictions.append(pred)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        
        # Return most common prediction in window
        if self.predictions:
            return max(set(self.predictions), key=self.predictions.count)
        return pred

smoother = PredictionSmoother(window_size=5)

# =======================
# 6. WEBCAM LOOP
# =======================
print("🎥 Starting webcam...")
print("Press 'q' to quit")
cap = cv2.VideoCapture(0)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

frame_count = 0
confidence_threshold = 0.6  # Only show predictions with confidence > 60%

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame_count += 1
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Display instructions
    cv2.putText(frame, "GesturiX - ASL Live Translation", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Convert to coordinates
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            coords = normalize_landmarks(coords)
            features = torch.tensor(coords.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

            # Model prediction
            with torch.no_grad():
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probabilities, 1)
                
                # Only show prediction if confidence is high enough
                if confidence.item() > confidence_threshold:
                    pred_idx = pred.item()
                    label = class_names[pred_idx]
                    
                    # Smooth prediction
                    label = class_names[smoother.update(pred_idx)]
                    conf_percent = confidence.item() * 100
                else:
                    label = "?"
                    conf_percent = 0

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Draw label with background box
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h) - 40
            
            # Background box for text
            text = f"Sign: {label}"
            if conf_percent > 0:
                text += f" ({conf_percent:.0f}%)"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )
            
            # Draw semi-transparent background
            cv2.rectangle(frame, 
                         (x - 5, y - text_height - 10), 
                         (x + text_width + 5, y + 5),
                         (0, 0, 0), -1)
            
            # Draw text
            color = (0, 255, 0) if conf_percent > confidence_threshold * 100 else (0, 165, 255)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 2, cv2.LINE_AA)
    else:
        # No hands detected
        cv2.putText(frame, "No hands detected", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Show FPS
    if frame_count % 30 == 0:  # Update every 30 frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow("GesturiX Live Sign Translation", frame)

    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")