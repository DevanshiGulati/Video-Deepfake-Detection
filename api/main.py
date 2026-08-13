import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.inference import load_model, predict_from_frames_dir
from extract_single import process_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = PROJECT_ROOT / "uploaded_videos"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHTS = Path(
    os.getenv("MODEL_WEIGHTS", str(PROJECT_ROOT / "deepfake_model_best_v2.pth"))
).expanduser()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

app = FastAPI(title="Deepfake Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/frames", StaticFiles(directory=str(UPLOADS_DIR)), name="frames")

_model = None
_device = None


def get_model():
    global _model, _device
    if _model is None:
        try:
            _model, _device = load_model(str(MODEL_WEIGHTS))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to load model weights: {exc}",
            ) from exc
    return _model, _device


def validate_video(file: UploadFile) -> str:
    filename = (file.filename or "").strip()
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed: {allowed}",
        )
    return extension


async def save_upload(file: UploadFile, destination: Path):
    total = 0
    chunk_size = 1024 * 1024
    with destination.open("wb") as output:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit",
                )
            output.write(chunk)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "modelConfigured": MODEL_WEIGHTS.is_file(),
        "device": str(_device) if _device else "not_loaded",
    }


@app.get("/")
def root():
    return {"name": "Deepfake Detector API", "status": "running"}


@app.post("/api/process-video")
async def process_video_api(file: UploadFile = File(...)):
    extension = validate_video(file)
    uid = uuid.uuid4().hex
    out_dir = UPLOADS_DIR / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    uploaded = out_dir / f"{uid}{extension}"

    try:
        await save_upload(file, uploaded)
        if not process_video(uploaded, out_dir):
            raise HTTPException(status_code=500, detail="Frame extraction failed")

        frame_urls = [
            f"/frames/{uid}/{path.name}"
            for path in sorted(out_dir.glob("*.jpg"))
        ]
        return {
            "outputDir": f"/frames/{uid}",
            "numFrames": len(frame_urls),
            "frameUrls": frame_urls,
            "uploadId": uid,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}") from exc


@app.post("/api/predict-video")
async def predict_video(file: UploadFile = File(...)):
    extension = validate_video(file)
    uid = uuid.uuid4().hex
    out_dir = UPLOADS_DIR / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    uploaded = out_dir / f"{uid}{extension}"

    try:
        await save_upload(file, uploaded)
        if not process_video(uploaded, out_dir):
            raise HTTPException(status_code=500, detail="Frame extraction failed")

        model, device = get_model()
        result = predict_from_frames_dir(model, device, str(out_dir))
        picked = result.pop("pickedFrames", [])
        frame_urls = [f"/frames/{uid}/{name}" for name in picked]

        prediction = "fake" if result["pred"] == 1 else "real"
        confidence = float(result["probs"][result["pred"]])

        return {
            **result,
            "prediction": prediction,
            "confidence": confidence,
            "frameUrls": frame_urls,
            "labelMap": {"0": "real", "1": "fake"},
            "message": "Video analysis complete",
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {exc}") from exc
