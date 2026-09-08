# MediaDNA — GUI for OpenAVFF Deepfake Detection

Local web application for video deepfake detection powered by OpenAVFF.

## Architecture

```
Frontend (React + TypeScript + Vite + Tailwind)
    ↓ HTTP API
Backend (FastAPI + Python)
    ↓
OpenAVFF Inference Service
    ↓
VideoCAVMAEFT Model (PyTorch, GPU)
```

## Quick Start

### Prerequisites

- **Conda environment**: `avff` (with PyTorch, torchaudio, decord, etc.)
- **Node.js**: v18+ (for frontend)
- **FFmpeg**: Already configured at `C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\...`
- **GPU**: NVIDIA RTX 4070 with CUDA

### Terminal 1 — Backend

```cmd
conda activate avff
cd C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The model loads on startup (~15 seconds). You'll see:
```
MediaDNA Backend Starting
Checkpoint: exp\stage-3-medium\models\best_audio_model.pth
Model loaded successfully!
```

### Terminal 2 — Frontend

```cmd
cd C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\frontend
npm run dev
```

### Open Browser

Navigate to: **http://localhost:5173**

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Backend and model status |
| `GET` | `/api/model-info` | Model configuration details |
| `POST` | `/api/analyze` | Analyze a video file (multipart upload) |

### Example: Health Check

```cmd
curl http://localhost:8000/api/health
```

### Example: Analyze Video

```cmd
curl -X POST -F "video=@path\to\video.mp4" http://localhost:8000/api/analyze
```

## Checkpoint

The GUI uses: `exp/stage-3-local/models/best_audio_model.pth`

This checkpoint was empirically selected over the `stage-3-medium` checkpoint 
after a comparative evaluation on 40 held-out videos (20 Fake, 20 Real) from FakeAVCeleb. 
While both models achieved high ROC-AUC, the `stage-3-local` model demonstrated far 
superior calibration at the default 0.5 threshold (90% accuracy vs 50% accuracy).

To use a different checkpoint, set the environment variable:
```cmd
set MEDIADNA_CHECKPOINT=exp\stage-3-medium\models\best_audio_model.pth
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Validation

Run the CLI vs GUI comparison script:

```cmd
conda activate avff
cd C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF
python compare_cli_gui.py
```

This validates that the GUI produces the same probabilities as direct CLI inference.

## Project Structure

```
OpenAVFF/
├── src/                          # Existing OpenAVFF model code (UNCHANGED)
├── checkpoints/                  # Pre-trained weights (UNCHANGED)
├── exp/                          # Trained checkpoints (UNCHANGED)
├── backend/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   ├── inference.py              # OpenAVFF inference service
│   ├── schemas.py                # Pydantic response models
│   └── requirements.txt          # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Main app with routing
│   │   ├── pages/                # Dashboard, Analyze, History, About
│   │   ├── components/           # Header, UploadZone, ResultCard, etc.
│   │   ├── api/                  # API client
│   │   ├── hooks/                # useHistory hook
│   │   └── types/                # TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts
├── compare_cli_gui.py            # Validation script
└── README_GUI.md                 # This file
```

## Label Semantics

- **Fake = 1** (output index 0 = Fake logit)
- **Real = 0** (output index 1 = Real logit)
- Threshold: P(Fake) >= 0.5 → "Likely Fake"
- These match the existing eval.py behavior exactly

## Disclaimer

AI-generated analysis is probabilistic and should not be treated as definitive
proof of authenticity. The model was trained on FakeAVCeleb and may not generalize
to all deepfake techniques.
