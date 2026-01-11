import os
import glob
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm

# Attention Module
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


# Model Architecture
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


val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeFinal().to(device).eval()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)  # ✅ CHANGED: Added strict=False
    return model, device

def build_clip_from_dir(frames_dir: str, num_frames: int = 16):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not paths:
        raise ValueError("No frames found")
    step = max(1, len(paths) // num_frames)
    idxs = list(range(0, len(paths), step))[:num_frames]
    imgs = [val_transform(Image.open(paths[i]).convert("RGB")) for i in idxs]
    if len(imgs) < num_frames:
        imgs += [imgs[-1]] * (num_frames - len(imgs))
    clip = torch.stack(imgs, dim=0).unsqueeze(0)  # (1,T,C,H,W)
    picked = [os.path.basename(paths[i]) for i in idxs]
    return clip, picked

def predict_from_frames_dir(model, device, frames_dir: str):
    x, picked = build_clip_from_dir(frames_dir)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred = int(torch.argmax(logits, dim=1).item())
    return {"pred": pred, "probs": probs, "pickedFrames": picked}
