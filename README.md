# Video Deepfake Detection

Full-stack video deepfake detection application using a React/Vite frontend, FastAPI backend, MediaPipe face extraction, and an EfficientNet-B3 + BiLSTM + Attention classifier.

## Architecture

`React/Vite → FastAPI → video validation → face/frame extraction → EfficientNet-B3 → BiLSTM → Attention → Real/Fake probabilities`

The frontend is in `frontend/` and the API is at the repository root.

## Backend

Python 3.11+ is required.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Set the model path with `MODEL_WEIGHTS`. If it is not set, the API looks for `deepfake_model_best_v2.pth` in the project root.

Start the API:

```bash
uvicorn api.main:app --reload --port 8000
```

Check it with `GET /healthz`.

## Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env` from `.env.example`:

```env
VITE_API_URL=http://localhost:8000
```

Start Vite:

```bash
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

## API

### `POST /api/predict-video`

Multipart form field:

- `file`: video (`mp4`, `mov`, `avi`, `mkv`, or `webm`), up to 100 MB by default.

The response contains the predicted class, both class probabilities, confidence, and URLs for sampled evidence frames.

## Model

The production inference architecture is:

- EfficientNet-B3 spatial feature extractor
- 16-frame temporal clip
- Bidirectional LSTM
- Self-attention over temporal features
- 2-class classifier (`real=0`, `fake=1`)

The checkpoint must match this architecture. Inference intentionally fails if the checkpoint contains missing or unexpected parameters instead of silently accepting an incompatible model.

## Development

A GitHub Actions workflow builds the frontend and compiles the Python backend on pushes and pull requests.
