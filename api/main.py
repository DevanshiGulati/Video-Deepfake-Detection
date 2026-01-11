from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from extract_single import process_video
import os
from api.inference import load_model, predict_from_frames_dir

app = FastAPI(title="Deepfake Pipeline API")

# Simpler CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Rest of your code remains the same...
uploads_dir = Path("uploaded_videos")
uploads_dir.mkdir(parents=True, exist_ok=True)

app.mount("/frames", StaticFiles(directory=str(uploads_dir)), name="frames")

MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "deepfake_model_best_v2.pth")
_model_cache = {"model": None, "device": None}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/api/process-video")
async def process_video_api(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files supported")

    uid = uuid.uuid4().hex
    out_dir = uploads_dir / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    uploaded = out_dir / f"{uid}.mp4"
    uploaded.write_bytes(await file.read())

    ok = process_video(uploaded, out_dir)
    if not ok:
        raise HTTPException(status_code=500, detail="Processing failed")

    jpgs = sorted(out_dir.glob("*.jpg"))
    frame_urls = [f"/frames/{out_dir.name}/{p.name}" for p in jpgs]

    return {
        "outputDir": f"/frames/{out_dir.name}",
        "numFrames": len(frame_urls),
        "frameUrls": frame_urls,
        "uploadId": uid,
    }

@app.post("/api/predict-video")
async def predict_video(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files supported")

    uid = uuid.uuid4().hex
    out_dir = uploads_dir / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    uploaded = out_dir / f"{uid}.mp4"
    uploaded.write_bytes(await file.read())

    ok = process_video(uploaded, out_dir)
    if not ok:
        raise HTTPException(status_code=500, detail="Frame extraction failed")

    if _model_cache["model"] is None:
        if not Path(MODEL_WEIGHTS).exists():
            raise HTTPException(status_code=500, detail=f"Weights not found: {MODEL_WEIGHTS}")
        _model_cache["model"], _model_cache["device"] = load_model(MODEL_WEIGHTS)

    result = predict_from_frames_dir(_model_cache["model"], _model_cache["device"], str(out_dir))
    picked = result.pop("pickedFrames", [])
    frame_urls = [f"/frames/{out_dir.name}/{name}" for name in picked]
    return {**result, "frameUrls": frame_urls, "labelMap": {"0": "real", "1": "fake"}}
