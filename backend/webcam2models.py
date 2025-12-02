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

model1_path = "backend/model1.pth"  # 0-9, a-z (36 classes)
model2_path = "backend/model2.pth"  # A-Z, del, nothing, space (29 classes)

# =======================
# 2. MODEL DEFINITIONS
# =======================
class SignNetModel1(nn.Module):
    """Model 1: 36 classes (0-9, a-z)"""
    def __init__(self, num_classes=36):
        super(SignNetModel1, self).__init__()
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

class SignNetModel2(nn.Module):
    """Model 2: 29 classes (A-Z, del, nothing, space)"""
    def __init__(self, num_classes=29):
        super(SignNetModel2, self).__init__()
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

# =======================
# 3. LOAD MODELS
# =======================
print("📂 Loading models...")

# Model 1: 0-9, a-z
model1 = SignNetModel1(num_classes=36).to(device)
try:
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    model1.eval()
    print("✅ Model 1 loaded (0-9, a-z)")
except FileNotFoundError:
    print(f"❌ Model 1 not found at: {model1_path}")
    model1 = None

# Model 2: A-Z, del, nothing, space
model2 = SignNetModel2(num_classes=29).to(device)
try:
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.eval()
    print("✅ Model 2 loaded (A-Z, del, nothing, space)")
except FileNotFoundError:
    print(f"❌ Model 2 not found at: {model2_path}")
    model2 = None

if model1 is None and model2 is None:
    print("❌ No models loaded. Exiting.")
    exit()

# =======================
# 4. CLASS NAMES
# =======================
# Model 1: 0-9, a-z
model1_classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]

# Model 2: A-Z, del, nothing, space
model2_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ['del', 'nothing', 'space']

# =======================
# 5. DUAL MODEL PREDICTOR
# =======================
class DualModelPredictor:
    """Intelligently combine predictions from both models"""
    def __init__(self, model1, model2, device):
        self.model1 = model1
        self.model2 = model2
        self.device = device
        
    def predict(self, features):
        """
        Predict using both models and combine intelligently
        Returns: (label, confidence, source_model)
        """
        predictions = []
        
        # Get prediction from Model 1 (0-9, a-z)
        if self.model1 is not None:
            with torch.no_grad():
                outputs1 = self.model1(features)
                probs1 = torch.softmax(outputs1, dim=1)
                conf1, pred1 = torch.max(probs1, 1)
                
                pred_idx1 = pred1.item()
                label1 = model1_classes[pred_idx1]
                confidence1 = conf1.item()
                
                predictions.append({
                    'label': label1,
                    'confidence': confidence1,
                    'model': 'Model 1',
                    'is_number': label1.isdigit()
                })
        
        # Get prediction from Model 2 (A-Z, del, nothing, space)
        if self.model2 is not None:
            with torch.no_grad():
                outputs2 = self.model2(features)
                probs2 = torch.softmax(outputs2, dim=1)
                conf2, pred2 = torch.max(probs2, 1)
                
                pred_idx2 = pred2.item()
                label2 = model2_classes[pred_idx2]
                confidence2 = conf2.item()
                
                predictions.append({
                    'label': label2,
                    'confidence': confidence2,
                    'model': 'Model 2',
                    'is_special': label2 in ['del', 'nothing', 'space']
                })
        
        # Decision logic
        if len(predictions) == 2:
            pred1, pred2 = predictions
            
            # If Model 1 predicts a number (0-9), use it (Model 2 doesn't have numbers)
            if pred1['is_number'] and pred1['confidence'] > 0.5:
                return pred1['label'], pred1['confidence'], pred1['model']
            
            # If Model 2 predicts special commands, use it (Model 1 doesn't have these)
            if pred2['is_special'] and pred2['confidence'] > 0.6:
                return pred2['label'], pred2['confidence'], pred2['model']
            
            # For letters (a-z vs A-Z), use the one with higher confidence
            # Convert to same case for comparison
            if pred1['label'].lower() == pred2['label'].lower():
                # Same letter detected by both, use higher confidence
                if pred1['confidence'] > pred2['confidence']:
                    return pred1['label'], pred1['confidence'], pred1['model']
                else:
                    return pred2['label'], pred2['confidence'], pred2['model']
            
            # Different predictions, use higher confidence
            if pred1['confidence'] > pred2['confidence']:
                return pred1['label'], pred1['confidence'], pred1['model']
            else:
                return pred2['label'], pred2['confidence'], pred2['model']
        
        elif len(predictions) == 1:
            # Only one model available
            pred = predictions[0]
            return pred['label'], pred['confidence'], pred['model']
        
        return None, 0.0, None

predictor = DualModelPredictor(model1, model2, device)

# =======================
# 6. MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# =======================
# 7. HELPER FUNCTIONS
# =======================
def normalize_landmarks(coords):
    """Normalize hand landmarks relative to wrist"""
    wrist = coords[0]
    coords -= wrist
    coords /= np.max(np.abs(coords) + 1e-6)
    return coords

# =======================
# 8. PREDICTION SMOOTHING
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
# 9. WEBCAM LOOP
# =======================
print("🎥 Starting webcam...")
print("Press 'q' to quit")
print("Press 'm' to toggle model display")
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

frame_count = 0
confidence_threshold = 0.5
show_model_info = True  # Toggle to show which model is being used

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = frame.shape

    # Display instructions
    cv2.putText(frame, "GesturiX - Dual Model ASL Translation", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit | 'm' to toggle model info", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Show active models
    models_text = "Active: "
    if model1: models_text += "Model 1 (0-9,a-z) "
    if model2: models_text += "Model 2 (A-Z,del,space) "
    cv2.putText(frame, models_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Convert to coordinates
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            coords = normalize_landmarks(coords)
            features = torch.tensor(coords.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

            # Get prediction from both models
            label, confidence, source_model = predictor.predict(features)
            
            if label and confidence > confidence_threshold:
                # Smooth prediction
                smoothed_label = label  # You can add smoothing if needed
                conf_percent = confidence * 100
            else:
                smoothed_label = "?"
                conf_percent = 0
                source_model = None

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Draw label with background box
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h) - 40
            
            # Build text
            text = f"Sign: {smoothed_label}"
            if conf_percent > 0:
                text += f" ({conf_percent:.0f}%)"
            
            if show_model_info and source_model:
                model_text = f"[{source_model}]"
                cv2.putText(frame, model_text, (x, y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )
            
            # Draw background
            cv2.rectangle(frame, 
                         (x - 5, y - text_height - 10), 
                         (x + text_width + 5, y + 5),
                         (0, 0, 0), -1)
            
            # Draw text with color based on source
            if source_model == "Model 1":
                color = (0, 255, 255)  # Yellow for Model 1
            elif source_model == "Model 2":
                color = (255, 0, 255)  # Magenta for Model 2
            else:
                color = (0, 165, 255)  # Orange for uncertain
                
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No hands detected", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Show FPS
    if frame_count % 30 == 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("GesturiX Dual Model Live Translation", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        show_model_info = not show_model_info
        print(f"Model info display: {'ON' if show_model_info else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")