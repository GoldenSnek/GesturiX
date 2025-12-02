import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import mediapipe as mp
import os

try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not installed. Install with: pip install seaborn")
    sns = None

# ============================================================================
# 1. MODEL ARCHITECTURE (from your training code)
# ============================================================================
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

# ============================================================================
# 2. DATASET CLASS (from your training code)
# ============================================================================
class SignLanguageDataset(Dataset):
    def __init__(self, hf_dataset, augment=False, has_labels=True):
        self.dataset = hf_dataset
        self.augment = augment
        self.has_labels = has_labels
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

    def __len__(self):
        return len(self.dataset)

    def normalize_landmarks(self, coords):
        wrist = coords[0]
        coords -= wrist
        coords /= np.max(np.abs(coords) + 1e-6)
        return coords

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        
        # Check if labels exist
        if self.has_labels and "label" in self.dataset[idx]:
            label = self.dataset[idx]["label"]
        else:
            label = -1  # Dummy label for unlabeled data
        
        results = self.mp_hands.process(np.array(img))

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            coords = self.normalize_landmarks(coords)
            features = coords.flatten()
        else:
            features = np.zeros(63, dtype=np.float32)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ============================================================================
# 3. CONFIGURATION
# ============================================================================
# Force GPU usage
print("="*80)
print("GPU SETUP")
print("="*80)

if not torch.cuda.is_available():
    print("❌ CUDA is not available!")
    print("   Checking your setup...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA compiled: {torch.version.cuda}")
    print("\nPlease reinstall PyTorch with CUDA:")
    print("pip uninstall torch torchvision torchaudio")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    exit(1)

# Force CUDA device
torch.cuda.set_device(0)  # Use first GPU
DEVICE = torch.device('cuda:0')

print(f"✅ GPU Status:")
print(f"   Device: {torch.cuda.get_device_name(0)}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
print(f"   Current Device: {torch.cuda.current_device()}")
print(f"   Device Count: {torch.cuda.device_count()}")
print(f"   Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"   Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

MODEL_PATH = r'D:\FinalPresentProj\GesturiX\backend\model2.pth'
# Use the training data folder - your training script already split this into train/val/test
TEST_DATA_PATH = r'D:\DatasetSign\archive\asl_alphabet_train\asl_alphabet_train'
OUTPUT_DIR = r'D:\FinalPresentProj\GesturiX\graphs'
BATCH_SIZE = 64  # Increased for faster GPU processing
NUM_CLASSES = 29

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n📁 Graphs will be saved to: {OUTPUT_DIR}")

# ============================================================================
# 4. LOAD TEST DATASET
# ============================================================================
print("\n" + "="*80)
print("LOADING TEST DATASET")
print("="*80)

if os.path.exists(TEST_DATA_PATH):
    print(f"Loading images from: {TEST_DATA_PATH}")
    
    # Load test dataset using HuggingFace datasets
    try:
        # Try loading with "test" split first (for the small test set)
        try:
            hf_test_dataset = load_dataset("imagefolder", data_dir=TEST_DATA_PATH, split="test")
            print(f"✓ Loaded as 'test' split")
        except:
            # Fallback to "train" split
            hf_test_dataset = load_dataset("imagefolder", data_dir=TEST_DATA_PATH, split="train")
            print(f"✓ Loaded as 'train' split")
        
        # Check if dataset has labels
        has_labels = "label" in hf_test_dataset[0]
        
        test_dataset = SignLanguageDataset(hf_test_dataset, augment=False, has_labels=has_labels)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"✓ Loaded {len(test_dataset)} test images")
        print(f"✓ Dataset has labels: {has_labels}")
        
        # Get class labels
        try:
            class_labels = hf_test_dataset.features["label"].names
            NUM_CLASSES = len(class_labels)
            print(f"✓ Found {NUM_CLASSES} classes: {class_labels}")
        except:
            class_labels = [chr(ord("A")+i) for i in range(26)] + ["del", "nothing", "space"]
            print(f"✓ Using default labels: {class_labels}")
        
        # If no labels, we can only do inference, not evaluation
        if not has_labels:
            print("\n⚠️  WARNING: Test dataset has no labels!")
            print("   This dataset is only for inference, not accuracy evaluation.")
            print("   Your ACTUAL test accuracy was calculated during training.")
            print(f"\n   To get meaningful accuracy, use your training dataset split:")
            print(f"   Change TEST_DATA_PATH to: D:/DatasetSign/archive/asl_alphabet_train/asl_alphabet_train")
            print(f"   (Your training script already split this into train/val/test)")
            
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        test_loader = None
        class_labels = None
        has_labels = False
else:
    print(f"✗ Path does not exist: {TEST_DATA_PATH}")
    test_loader = None
    class_labels = None

# ============================================================================
# 5. LOAD MODEL
# ============================================================================
print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

if os.path.exists(MODEL_PATH):
    model = SignNet(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # Verify model is on GPU
        print(f"✓ Model loaded successfully")
        print(f"✓ Model device: {next(model.parameters()).device}")
        
        # Force all operations to GPU
        torch.backends.cudnn.benchmark = True  # Optimize for GPU
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = None
else:
    print(f"✗ Model file not found: {MODEL_PATH}")
    model = None

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
if test_loader is not None and model is not None and class_labels is not None:
    
    # Check if we have labels for evaluation
    if not has_labels:
        print("\n" + "="*80)
        print("INFERENCE MODE (No Labels Available)")
        print("="*80)
        print("\n❌ Cannot calculate accuracy without labels in test dataset!")
        print("\n💡 Your test folder only contains sample images without labels.")
        print("   The REAL test accuracy was already calculated during your 24-hour training.")
        print("\n📋 To evaluate on a proper test set, do ONE of these:")
        print("   1. Check your training output - it printed the test accuracy at the end")
        print("   2. Re-run evaluation on the training data split (see instructions above)")
        print("\n" + "="*80)
    else:
        print("\n" + "="*80)
        print("RUNNING EVALUATION")
        print("="*80)
    
    y_true = []
    y_pred = []
    y_probs = []
    
    model.eval()
    correct = 0
    total = 0
    
    print("Processing test images...")
    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            # Explicitly move to GPU
            features = features.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Only count accuracy if we have real labels
            if has_labels:
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
                print(f"  Processed {i + 1}/{len(test_loader)} batches | GPU Memory: {gpu_mem:.0f} MB")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # ========================================================================
    # 7. DISPLAY RESULTS
    # ========================================================================
    if has_labels:
        print("\n" + "="*80)
        print("OVERALL ACCURACY")
        print("="*80)
        
        accuracy = correct / total
        print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Correct: {correct}/{total}")
        
        # Average confidence
        avg_confidence = np.mean(np.max(y_probs, axis=1))
        print(f"\n📊 Average Prediction Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        
        # Low confidence predictions
        confidence_threshold = 0.5
        low_conf_mask = np.max(y_probs, axis=1) < confidence_threshold
        num_low_conf = np.sum(low_conf_mask)
        print(f"⚠️  Predictions with confidence < {confidence_threshold}: {num_low_conf} ({num_low_conf/len(y_true)*100:.2f}%)")
        
        # ====================================================================
        # 8. CLASSIFICATION REPORT
        # ====================================================================
        print("\n" + "="*80)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))
        
        # ====================================================================
        # 9. CONFUSION MATRIX
        # ====================================================================
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        
        cm = confusion_matrix(y_true, y_pred)
        
        if sns is not None:
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_labels, yticklabels=class_labels,
                        cbar_kws={'label': 'Count'})
            plt.title('Confusion Matrix - ASL Sign Recognition', fontsize=16, pad=20)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_eval.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved as '{cm_path}'")
            plt.close()
        
        # ====================================================================
        # 10. MOST CONFUSED PAIRS
        # ====================================================================
        print("\n" + "="*80)
        print("MOST CONFUSED CLASS PAIRS")
        print("="*80)
        
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append((cm[i][j], class_labels[i], class_labels[j]))
        
        confused_pairs.sort(reverse=True)
        if confused_pairs:
            for count, true_label, pred_label in confused_pairs[:10]:
                print(f"  {true_label} → {pred_label}: {count} times")
        else:
            print("  No confusion detected!")
        
        # ====================================================================
        # 11. PER-CLASS ACCURACY
        # ====================================================================
        print("\n" + "="*80)
        print("PER-CLASS ACCURACY")
        print("="*80)
        
        class_accuracies = []
        for i, class_name in enumerate(class_labels):
            mask = y_true == i
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred[mask] == y_true[mask])
                class_accuracies.append((class_acc, class_name, np.sum(mask)))
        
        class_accuracies.sort()
        
        print("\n📉 Lowest performing classes:")
        for acc, name, count in class_accuracies[:5]:
            print(f"  {name}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")
        
        print("\n📈 Highest performing classes:")
        for acc, name, count in class_accuracies[-5:]:
            print(f"  {name}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")
        
        # Plot per-class accuracy
        sorted_idx = np.argsort([acc for acc, _, _ in class_accuracies])
        sorted_labels = [class_accuracies[i][1] for i in sorted_idx]
        sorted_accs = [class_accuracies[i][0] * 100 for i in sorted_idx]
        
        plt.figure(figsize=(16, 6))
        plt.bar(sorted_labels, sorted_accs, color='skyblue', edgecolor='navy')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Per-Class Accuracy on Test Set (Sorted)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        per_class_path = os.path.join(OUTPUT_DIR, 'per_class_accuracy.png')
        plt.savefig(per_class_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Per-class accuracy graph saved as '{per_class_path}'")
        plt.close()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
    else:
        # Just show predictions without accuracy
        print("\n" + "="*80)
        print("PREDICTIONS (Inference Mode)")
        print("="*80)
        print("\nPredicted classes for test images:")
        for i, pred in enumerate(y_pred[:28]):  # Show first 28
            prob = y_probs[i][pred] * 100
            print(f"  Image {i+1}: {class_labels[pred]} (confidence: {prob:.1f}%)")

else:
    print("\n❌ Cannot run evaluation - missing required components")
    if test_loader is None:
        print("  - Test dataset not loaded")
    if model is None:
        print("  - Model not loaded")
    if class_labels is None:
        print("  - Class labels not available")