from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

SEQ_LEN = 32
CROP_SIZE = 224
MARGIN = 20


def sample_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return np.linspace(0, n - 1, k).astype(int).tolist()


def read_frames(video_path: Path):
    import decord
    return decord.VideoReader(str(video_path))


def detect_largest_box(rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection
    # Recreate detector each call is expensive; create one and stash on function
    if not hasattr(detect_largest_box, "_detector"):
        detect_largest_box._detector = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    detector = detect_largest_box._detector

    res = detector.process(rgb)
    if not res.detections:
        return None
    H, W, _ = rgb.shape
    best = None
    best_area = -1
    for d in res.detections:
        bb = d.location_data.relative_bounding_box
        x1 = int(bb.xmin * W)
        y1 = int(bb.ymin * H)
        w = int(bb.width * W)
        h = int(bb.height * H)
        area = w * h
        if area > best_area:
            best_area = area
            best = (x1, y1, w, h)
    return best


def crop_with_margin(arr: np.ndarray, box: Optional[Tuple[int, int, int, int]], margin: int = MARGIN) -> Optional[np.ndarray]:
    if box is None:
        return None
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    H, W = arr.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W, x2 + margin)
    y2 = min(H, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return arr[y1:y2, x1:x2]


def process_video(src: Path, dst_dir: Path) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    vr = read_frames(src)
    idxs = sample_indices(len(vr), SEQ_LEN)
    batch = vr.get_batch(idxs).asnumpy()  # (T,H,W,3) RGB
    for i, frame in enumerate(batch):
        box = detect_largest_box(frame)
        crop = crop_with_margin(frame, box, margin=MARGIN) if box else None
        if crop is None:
            H, W, _ = frame.shape
            s = min(H, W)
            y0 = (H - s) // 2
            x0 = (W - s) // 2
            crop = frame[y0:y0 + s, x0:x0 + s]
        Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR).save(
            dst_dir / f"{i:04d}.jpg", quality=95
        )
    return True


def main():
    cwd = Path.cwd()
    videos = sorted(cwd.glob("*.mp4"))
    if not videos:
        print("No MP4 file found in", cwd)
        return
    # Take the first MP4
    src = videos[0]
    out_dir = cwd / src.stem
    print(f"Processing {src.name} -> {out_dir}")
    ok = process_video(src, out_dir)
    if ok:
        print("Done.")
    else:
        print("Failed to process video.")


if __name__ == "__main__":
    main()

