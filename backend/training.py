import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

# =======================
# 1. CONFIG
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

num_classes = 36
batch_size = 64
epochs = 10 # adjust as needed
lr = 0.0001
model_name = "model1"
# =======================
# 2. MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

# =======================
# 3. DATASET CLASS
# =======================
class SignLanguageDataset(Dataset):
    def __init__(self, hf_dataset, augment=False):
        self.dataset = hf_dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def normalize_landmarks(self, coords):
        wrist = coords[0]
        coords -= wrist
        coords /= np.max(np.abs(coords) + 1e-6)
        return coords

    def augment_landmarks(self, coords):
        coords += np.random.normal(0, 0.02, coords.shape)
        scale = random.uniform(0.9, 1.1)
        coords *= scale
        if random.random() > 0.5:
            coords[:,0] *= -1
        return coords

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]

        results = mp_hands.process(np.array(img))
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            coords = self.normalize_landmarks(coords)
            if self.augment:
                coords = self.augment_landmarks(coords)
            features = coords.flatten()
        else:
            features = np.zeros(63, dtype=np.float32)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# =======================
# 4. LOAD DATASET AND SPLIT
# =======================
print("📂 Loading dataset...")
hf_dataset = load_dataset("Hemg/sign_language_dataset")["train"]

# Split into 70% train, 15% val, 15% test
train_size = int(0.7 * len(hf_dataset))
val_size = int(0.15 * len(hf_dataset))
test_size = len(hf_dataset) - train_size - val_size
train_raw, val_raw, test_raw = random_split(hf_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_dataset = SignLanguageDataset(train_raw, augment=True)
val_dataset = SignLanguageDataset(val_raw, augment=False)
test_dataset = SignLanguageDataset(test_raw, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# =======================
# 5. MODEL
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# =======================
# 6. TRAINING LOOP
# =======================
train_accs, val_accs, train_losses, val_losses = [], [], [], []

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# =======================
# 7. SAVE MODEL
# =======================
torch.save(model.state_dict(), f"{model_name}.pth")
print(f"✅ Model saved as {model_name}.pth")

# =======================
# 8. PLOTS
# =======================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig(f"training_results_{model_name}.png")
print(f"✅ Training graphs saved as training_results_{model_name}.png")

# =======================
# 9. CONFUSION MATRIX
# =======================
all_labels, all_preds = [], []
model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(num_classes)])
disp.plot(xticks_rotation=90, cmap="Blues")
plt.savefig(f"confusion_matrix_{model_name}.png")
print(f"✅ Confusion matrix saved as confusion_matrix_{model_name}.png")

# =======================
# 10. PER-CLASS ACCURACY GRAPH
# =======================
class_labels = [str(i) for i in range(10)] + [chr(ord("a")+i) for i in range(26)]

per_class_correct = np.zeros(num_classes)
per_class_total = np.zeros(num_classes)

for i in range(len(all_labels)):
    label = all_labels[i]
    pred = all_preds[i]
    per_class_total[label] += 1
    if label == pred:
        per_class_correct[label] += 1

per_class_acc = per_class_correct / (per_class_total + 1e-6) * 100  # percentage

# Sort by accuracy
sorted_idx = np.argsort(per_class_acc)
sorted_labels = [class_labels[i] for i in sorted_idx]
sorted_acc = per_class_acc[sorted_idx]

plt.figure(figsize=(14,6))
plt.bar(sorted_labels, sorted_acc, color="skyblue")
plt.xlabel("Classes")
plt.ylabel("Accuracy (%)")
plt.title("Per-Class Accuracy on Test Set (Sorted)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"per_class_accuracy_{model_name}.png")
print(f"✅ Per-class accuracy graph saved as per_class_{model_name}.png")
