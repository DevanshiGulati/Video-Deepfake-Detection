"""
FINAL MODEL: EfficientNet-B3 + BiLSTM + Attention + 16 Frames
Syntactically fixed training script (local paths; no Colab mounts).
"""

import os
import glob
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm


warnings.filterwarnings("ignore")


# 1) DATASET (16 FRAMES)
class VideoDataset(Dataset):
    def __init__(self, root: str, transform=None, num_frames: int = 16) -> None:
        self.samples: List[Tuple[str, int]] = []
        self.num_frames = num_frames
        self.transform = transform

        for label, lbl in [("real", 0), ("fake", 1)]:
            dir_path = os.path.join(root, label)
            if not os.path.exists(dir_path):
                continue
            for vid in os.listdir(dir_path):
                vid_path = os.path.join(dir_path, vid)
                frames = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
                if len(frames) >= 32:
                    self.samples.append((vid_path, lbl))

        print(f"Found {len(self.samples)} videos")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        vid_path, label = self.samples[idx]
        all_frames = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
        step = max(1, len(all_frames) // self.num_frames)
        indices = list(range(0, len(all_frames), step))[: self.num_frames]
        imgs = [Image.open(all_frames[i]).convert("RGB") for i in indices]
        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]
        return torch.stack(imgs), label


# 2) AUGMENTATION
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# 3) ATTENTION MODULE
class Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        return torch.matmul(attn, v)


# 4) FINAL MODEL
class DeepfakeFinal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # returns features when num_classes=0
        self.backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)

        # Unfreeze last 4 stages
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ["blocks.3", "blocks.4", "blocks.5", "blocks.6"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.lstm = nn.LSTM(1536, 512, batch_first=True, bidirectional=True)
        self.attention = Attention(1024)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.backbone(x)
        x = x.view(b, t, -1)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = x.mean(dim=1)
        return self.fc(x)


if __name__ == "__main__":
    # 5) DATA LOADER
    # Update this path to your local dataset of frames: <root>/{real|fake}/{video_stem}/*.jpg
    root = "data/faces"

    full_ds = VideoDataset(root, transform=val_transform, num_frames=16)
    if len(full_ds) == 0:
        print("No samples found. Please ensure dataset exists at:", root)
        raise SystemExit(0)

    # Example split (adjust sizes to your dataset)
    n_total = len(full_ds)
    n_val = max(1, int(0.1 * n_total))
    n_test = max(1, int(0.1 * n_total))
    n_train = max(1, n_total - n_val - n_test)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform

    batch_size = 4
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 6) TRAINING
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeFinal().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    patience = 5
    wait = 0
    best_val = 0.0
    epochs = 2  # keep small for sanity run

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total_loss = 0.0
        for X, y in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            correct += (preds.argmax(1) == y).sum().item()
            total_loss += loss.item()
        train_acc = correct / len(train_ds)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                val_correct += (preds.argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_ds)

        scheduler.step(val_acc)
        print(
            f"Epoch {epoch:2d} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
            f"Loss: {total_loss/ max(1, len(train_dl)) :.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_acc > best_val:
            best_val = val_acc
            wait = 0
            torch.save(model.state_dict(), "deepfake_model_best_v2.pth")
            print(f"  [BEST] SAVED: {val_acc:.3f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 7) FINAL TEST
    model.load_state_dict(torch.load("deepfake_model_best_v2.pth", map_location=device))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            test_correct += (preds.argmax(1) == y).sum().item()
    test_acc = test_correct / len(test_ds)
    print(f"\n[FINAL] TEST ACCURACY: {test_acc:.2%}")
    print("Final model saved as deepfake_model_best_v2.pth")