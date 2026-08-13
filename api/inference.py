import glob
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


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
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        return torch.matmul(self.softmax(scores), v)


class DeepfakeFinal(nn.Module):
    """EfficientNet-B3 + BiLSTM + temporal self-attention classifier."""

    def __init__(self) -> None:
        super().__init__()
        # Checkpoints contain the complete backbone, so no ImageNet download is
        # needed during inference.
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=0
        )
        self.lstm = nn.LSTM(
            1536, 512, batch_first=True, bidirectional=True
        )
        self.attention = Attention(1024)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = x.shape
        x = x.reshape(batch * frames, channels, height, width)
        x = self.backbone(x)
        x = x.reshape(batch, frames, -1)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = x.mean(dim=1)
        return self.fc(x)


VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


def _normalize_checkpoint_keys(state: dict) -> dict:
    """Normalize legacy checkpoint naming to the canonical inference model."""
    normalized = {}
    for key, value in state.items():
        # training_example.py saved the attention module as ``attn``.
        # The canonical inference model uses ``attention``.
        if key.startswith("attn."):
            key = "attention." + key[len("attn."):]
        normalized[key] = value
    return normalized


def load_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(weights_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")

    model = DeepfakeFinal().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a model state_dict")

    state = _normalize_checkpoint_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch. "
            f"Missing keys: {missing[:5]} | Unexpected keys: {unexpected[:5]}"
        )

    model.eval()
    return model, device


def build_clip_from_dir(frames_dir: str, num_frames: int = 16):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not paths:
        raise ValueError("No extracted frames found")

    # Match the training convention: evenly sample the complete sequence.
    if len(paths) <= num_frames:
        indices = list(range(len(paths)))
    else:
        indices = torch.linspace(0, len(paths) - 1, num_frames).long().tolist()

    images = [
        VAL_TRANSFORM(Image.open(paths[index]).convert("RGB"))
        for index in indices
    ]

    if len(images) < num_frames:
        images.extend([images[-1]] * (num_frames - len(images)))

    clip = torch.stack(images).unsqueeze(0)
    picked = [os.path.basename(paths[index]) for index in indices]
    return clip, picked


def predict_from_frames_dir(model, device, frames_dir: str):
    x, picked = build_clip_from_dir(frames_dir)
    x = x.to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred = int(torch.argmax(logits, dim=1).item())

    return {
        "pred": pred,
        "probs": probs,
        "pickedFrames": picked,
    }
