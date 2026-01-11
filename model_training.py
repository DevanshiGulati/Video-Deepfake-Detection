
# 0. Install
!pip install -q timm

# 1. Imports & Drive mount
import os, glob, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# 2. Config — change these if needed
ROOT = "/content/all_data/faces"   # <-- dataset root (contains 'real' and 'fake' subfolders)
SAVE_PATH = "/content/drive/MyDrive/deepfake_faceforensics_final.pth"
NUM_FRAMES = 16      # frames per sample used for model input
BATCH_SIZE = 10
EPOCHS = 50
LR = 5e-5
WEIGHT_DECAY = 1e-2
USE_CLASS_WEIGHTS = False   # set True if dataset highly imbalanced
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# 3. Dataset class — supports folders with >= 32 frames (samples evenly)
class VideoDataset(Dataset):
    def __init__(self, root, transform=None, num_frames=16):
        self.samples = []
        self.num_frames = num_frames
        root = Path(root)
        for label_name, lbl in [("real", 0), ("fake", 1)]:
            dir_path = root / label_name
            if not dir_path.exists():
                continue
            for vid in sorted(os.listdir(dir_path)):
                vid_path = dir_path / vid
                if not vid_path.is_dir():
                    continue
                frames = sorted(glob.glob(str(vid_path / "*.jpg")))
                # accept folders that have at least num_frames (your data contains 32)
                if len(frames) >= self.num_frames:
                    self.samples.append((frames, lbl))
        self.transform = transform
        print(f"✅ Found {len(self.samples)} videos (real+fake) in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        total = len(frame_paths)
        # sample evenly NUM_FRAMES from available frames
        if total == self.num_frames:
            indices = list(range(total))
        else:
            # evenly spaced selection
            step = total / float(self.num_frames)
            indices = [int(step * i) for i in range(self.num_frames)]
        imgs = []
        for i in indices:
            p = frame_paths[i]
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        # returns tensor shape (T, C, H, W)
        return torch.stack(imgs), label

# 4. Transforms (same as your val/train transforms)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 5. Model: EfficientNet-B3 + BiLSTM + Attention
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        return torch.matmul(attn, V)

class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone: remove classifier head (num_classes=0 yields features)
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        # fine-tune last blocks only (option)
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ["blocks.4", "blocks.5", "blocks.6"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.lstm = nn.LSTM(1536, 512, batch_first=True, bidirectional=True)
        self.attn = Attention(1024)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)           # (B*T, feat=1536)
        x = x.view(B, T, -1)           # (B, T, 1536)
        x, _ = self.lstm(x)            # (B, T, 1024)
        x = self.attn(x)               # (B, T, 1024)
        x = x.mean(dim=1)              # (B, 1024)
        return self.fc(x)

# 6. Prepare dataset + loaders
full_ds = VideoDataset(ROOT, transform=None, num_frames=NUM_FRAMES)
n = len(full_ds)
if n == 0:
    raise SystemExit("No videos found. Update ROOT to correct path and rerun.")

train_size = int(0.8 * n)
val_size = int(0.1 * n)
test_size = n - (train_size + val_size)
train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size])

# attach transforms
train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform
test_ds.dataset.transform = val_transform

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Dataset sizes → total: {n}, train: {train_size}, val: {val_size}, test: {test_size}")

# 7. Loss, optimizer, scheduler (optionally class weights if imbalance)
device = DEVICE
model = DeepfakeModel().to(device)

if USE_CLASS_WEIGHTS:
    # compute class counts from dataset
    counts = [0, 0]
    for _, lbl in full_ds.samples:
        counts[lbl] += 1
    weights = [sum(counts)/c if c>0 else 1.0 for c in counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using class weights:", weights)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# 8. Training loop
best_val = 0.0
wait = 0
patience = 10

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    correct = 0
    nb = 0
    for X, y in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)          # preds shape [B, 2]
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (preds.argmax(1) == y).sum().item()
        nb += X.size(0)
    train_acc = correct / nb
    # validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for Xv, yv in val_dl:
            Xv, yv = Xv.to(device), yv.to(device)
            val_correct += (model(Xv).argmax(1) == yv).sum().item()
    val_acc = val_correct / len(val_ds)
    scheduler.step(val_acc)
    print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {total_loss/len(train_dl):.4f}")
    # save best
    if val_acc > best_val:
        best_val = val_acc
        wait = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ [BEST] Saved model with val_acc={val_acc:.4f}")
    else:
        wait += 1
        if wait >= patience:
            print("⏹ Early stopping")
            break

# 9. Final evaluation on test set
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()
test_correct = 0
with torch.no_grad():
    for X, y in test_dl:
        X, y = X.to(device), y.to(device)
        test_correct += (model(X).argmax(1) == y).sum().item()
test_acc = test_correct / len(test_ds)
print(f"\n✅ FINAL TEST ACCURACY: {test_acc:.2%}")
print(f"Model saved to: {SAVE_PATH}")