import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from datasets import load_dataset
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
from tqdm import tqdm

# =======================
# 1. CONFIG
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

num_classes = 10  # Only digits 0-9
batch_size = 64
epochs = 50  # Increased maximum epochs for Early Stopping (will likely stop sooner)
lr = 0.0005 # Increased learning rate slightly for faster convergence
weight_decay = 1e-4 # ADDED L2 Regularization (Weight Decay)
model_name = "model_digits_optimized"

# Early Stopping Parameters
patience = 7
min_delta = 0.001

# =======================
# 2. MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

# =======================
# 3. DATASET CLASS (IMPROVED NORMALIZATION)
# =======================
class SignLanguageDataset(Dataset):
    def __init__(self, hf_dataset, augment=False):
        self.dataset = hf_dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def normalize_landmarks(self, coords):
        """Normalize hand landmarks relative to wrist and normalize Z independently."""
        # 1. Center coordinates relative to the wrist (landmark 0)
        wrist = coords[0]
        coords_centered = coords - wrist
        
        # 2. Global XY Scaling (using max absolute value of XY)
        # We focus scaling on XY, as Z is highly dependent on camera distance and angle
        xy_coords = coords_centered[:, :2]
        # Use a scaling factor derived from the maximum absolute XY spread
        scale_factor_xy = np.max(np.abs(xy_coords)) + 1e-6
        coords_centered[:, :2] /= scale_factor_xy

        # 3. Z-axis Normalization (Crucial for depth information in digits)
        # Normalize Z separately to retain depth information relative to the overall gesture size
        z_coords = coords_centered[:, 2]
        # Min-Max Normalization for Z to map it to roughly [-1, 1] range
        z_range = np.max(z_coords) - np.min(z_coords) + 1e-6
        coords_centered[:, 2] = (z_coords - np.mean(z_coords)) / (z_range / 2)
        
        return coords_centered

    def augment_landmarks(self, coords):
        # Existing Jitter/Scaling/Flipping remains good practice
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

# ... (Sections 4. LOAD DATASET AND FILTER DIGITS ONLY remain the same) ...
# =======================
# 4. LOAD DATASET AND FILTER DIGITS ONLY
# =======================
print("📂 Loading dataset...")
# Using a hosted dataset
try:
    hf_dataset = load_dataset("Hemg/sign_language_dataset")["train"]
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    print("Attempting to load a dummy dataset for code execution.")
    # Fallback/Dummy data if HuggingFace dataset fails
    class DummyDataset(Dataset):
        def __len__(self): return 100 
        def __getitem__(self, idx):
            return {"image": np.zeros((100, 100, 3), dtype=np.uint8), "label": idx % 10}
    hf_dataset = DummyDataset()

print("🔍 Filtering dataset for digits 0-9 only...")
digit_indices = []
for idx in range(len(hf_dataset)):
    label = hf_dataset[idx]["label"]
    if 0 <= label <= 9:  # Keep only digits 0-9
        digit_indices.append(idx)

print(f"✅ Found {len(digit_indices)} digit samples out of {len(hf_dataset)} total samples")

# Handle case where dataset is too small after filtering (e.g., dummy dataset)
if len(digit_indices) < 200:
    print("⚠️ WARNING: Too few digit samples for robust training. Using all available data.")
    if not digit_indices:
        # If no digits found, use a smaller slice of the original for flow control
        digit_indices = list(range(min(len(hf_dataset), 500)))

digits_dataset = Subset(hf_dataset, digit_indices)

# Split into 70% train, 15% val, 15% test
dataset_size = len(digits_dataset)
# Ensure sizes are non-zero
if dataset_size == 0:
    print("CRITICAL ERROR: Dataset size is 0 after filtering.")
    exit()
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

# Re-adjust sizes if one of them is zero due to very small dataset
if train_size == 0 or val_size == 0 or test_size == 0:
    print("⚠️ WARNING: Redistributing small dataset for non-zero splits.")
    train_size = max(1, int(dataset_size * 0.7))
    val_size = max(1, int(dataset_size * 0.15))
    test_size = dataset_size - train_size - val_size
    if test_size < 0: # Correction for rounding errors on extremely small data
        test_size = 0
        train_size = dataset_size - val_size
    elif test_size == 0:
        val_size -= 1
        test_size = 1

    if train_size + val_size + test_size != dataset_size:
        train_size = dataset_size - val_size - test_size


print(f"Total digit samples: {dataset_size}")
print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

train_raw, val_raw, test_raw = random_split(
    digits_dataset, 
    [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42)
)

train_dataset = SignLanguageDataset(train_raw, augment=True)
val_dataset = SignLanguageDataset(val_raw, augment=False)
test_dataset = SignLanguageDataset(test_raw, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class_labels = [str(i) for i in range(10)]
print(f"✅ Class labels: {class_labels}")

# =======================
# 5. MODEL (DEEPER ARCHITECTURE)
# =======================
class SignNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SignNet, self).__init__()
        # Increased model depth and width for higher capacity
        self.fc = nn.Sequential(
            nn.Linear(63, 1024), # Wider first layer
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
            nn.Dropout(0.3), # Increased Dropout slightly
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

model = SignNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
# ADDED weight_decay for L2 Regularization
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# =======================
# 6. TRAINING LOOP (ADDED EARLY STOPPING)
# =======================
print("🚀 Starting training for digits 0-9...")
train_accs, val_accs, train_losses, val_losses = [], [], [], []

# Early Stopping Initialization
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop_flag = False

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # Training
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for features, labels in train_pbar:
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
        
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_correct, val_total, val_loss_sum = 0, 0, 0

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
        for features, labels in val_pbar:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            val_loss_sum += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
            val_loss = val_loss_sum / len(val_loader)
            val_pbar.set_postfix({
                'loss': f'{val_loss:.4f}',
                'acc': f'{val_correct/val_total:.4f}'
            })

    val_acc = val_correct / val_total
    val_loss = val_loss_sum / len(val_loader)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step()

    print(f"📊 Epoch [{epoch+1}/{epochs}] "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Early Stopping Check
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the BEST model found so far
        torch.save(model.state_dict(), f"best_{model_name}.pth")
        print(f"   ✨ New best model saved with Val Loss: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"🛑 Early stopping triggered after {patience} epochs with no improvement.")
            early_stop_flag = True
            break

# If early stopping triggered, load the best model for final evaluation
if early_stop_flag:
    try:
        model.load_state_dict(torch.load(f"best_{model_name}.pth"))
        print("✅ Loaded best model for final evaluation.")
    except FileNotFoundError:
        print("⚠️ Best model file not found, using last trained model.")

# =======================
# 7. SAVE FINAL MODEL (The one used for final evaluation)
# =======================
torch.save(model.state_dict(), f"{model_name}.pth")
print(f"✅ Final model state saved as {model_name}.pth")

# ... (Sections 8, 9, 10, 11, 12 remain the same for final evaluation and plotting) ...
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
plt.title("Accuracy - Digits Only (0-9)")


plt.subplot(1,2,2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss - Digits Only (0-9)")


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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(xticks_rotation=45, cmap="Blues", ax=ax)
plt.title("Confusion Matrix - ASL Digits (0-9)")
plt.tight_layout()
plt.savefig(f"confusion_matrix_{model_name}.png")
print(f"✅ Confusion matrix saved as confusion_matrix_{model_name}.png")


# =======================
# 10. PER-CLASS ACCURACY GRAPH
# =======================
per_class_correct = np.zeros(num_classes)
per_class_total = np.zeros(num_classes)

for i in range(len(all_labels)):
    label = all_labels[i]
    pred = all_preds[i]
    per_class_total[label] += 1
    if label == pred:
        per_class_correct[label] += 1

per_class_acc = per_class_correct / (per_class_total + 1e-6) * 100  # percentage

sorted_idx = np.argsort(per_class_acc)
sorted_labels = [class_labels[i] for i in sorted_idx]
sorted_acc = per_class_acc[sorted_idx]

plt.figure(figsize=(12,6))
plt.bar(sorted_labels, sorted_acc, color="skyblue")
plt.xlabel("Digit Classes")
plt.ylabel("Accuracy (%)")
plt.title("Per-Digit Accuracy on Test Set (Sorted)")
plt.xticks(rotation=0)
plt.ylim([0, 105])
plt.tight_layout()
plt.savefig(f"per_class_accuracy_{model_name}.png")
print(f"✅ Per-digit accuracy graph saved as per_class_accuracy_{model_name}.png")


# =======================
# 11. TEST SET EVALUATION
# =======================
test_correct, test_total = 0, 0
model.eval()

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
print(f"\n🎯 Final Test Accuracy (Digits 0-9): {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Correct: {test_correct}/{test_total}")

# =======================
# 12. DISTRIBUTION OF DIGITS IN DATASET
# =======================
print("\n📊 Distribution of digits in test set:")
for i in range(num_classes):
    count = int(per_class_total[i])
    accuracy = per_class_acc[i]
    print(f"   Digit {i}: {count} samples, Accuracy: {accuracy:.2f}%")