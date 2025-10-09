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

num_classes = 36
model_path = "backend/model1.pth"

#change model path if needed

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

model = SignNet(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Class names (0-9, a-z)
class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]

# =======================
# 3. MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.7)

# =======================
# 4. HELPER FUNCTIONS
# =======================
def normalize_landmarks(coords):
    wrist = coords[0]
    coords -= wrist
    coords /= np.max(np.abs(coords) + 1e-6)
    return coords

# =======================
# 5. WEBCAM LOOP
# =======================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert to coordinates
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            coords = normalize_landmarks(coords)
            features = torch.tensor(coords.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

            # Model prediction
            with torch.no_grad():
                outputs = model(features)
                _, pred = torch.max(outputs, 1)
                label = class_names[pred.item()]

            # Draw landmarks and label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h) - 20
            cv2.putText(frame, f"Sign: {label}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("GesturiX Live Sign Translation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
