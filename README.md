# MediaDNA — Multimodal Deepfake Forensics

MediaDNA is an AI-powered digital forensics instrument built on the **OpenAVFF (VideoCAVMAEFT)** architecture. It detects synthetic media by analyzing spatial-temporal artifacts and acoustic-visual desynchronization.

*This project is a transparent research demonstration, not a production-grade automated moderation system. It exposes the limitations of modern multimodal attention networks alongside their successes.*

## Architecture Overview
MediaDNA operates on a two-pronged approach:
1. **OpenAVFF Neural Core**: A deep learning architecture that utilizes a Video-Audio Cross-Attention Masked Auto-Encoder. It rips out 16 uniform visual frames and a Kaldi log-mel audio filterbank, forcing the video to "attend" to the audio and vice versa to catch deepfake anomalies.
2. **Visual Heuristics (Explainability)**: A classical computer vision layer computing frame-to-frame Mean Absolute Error (MAE) and Laplacian variance to detect spatial blur and face-swap glitching without relying on a black-box neural net.

For full architectural details, see `docs/ARCHITECTURE.md`.

## Research & Metrics
The model was trained and evaluated on the FakeAVCeleb v1.2 dataset.
- **Baseline Accuracy**: 64.69%
- **Known Blindspot**: The model demonstrates a critical failure rate on **FakeVideo + RealAudio** permutations (27.20% accuracy), heavily over-indexing on pristine audio tracks.
- **Robustness**: The acoustic encoder completely fails under 10dB audio noise, but the visual encoder remains resilient to heavy CRF 40 video compression.

For full academic results, read `docs/RESEARCH_REPORT.md` and `docs/MODEL_CARD.md`.

## Project Structure
- `backend/`: FastAPI Python server handling file ingestion, ffmpeg extraction, PyTorch inference, and forensic archival.
- `frontend/`: React + Vite + Framer Motion cinematic user interface providing the Forensic Demo Lab.
- `docs/`: System architecture, research reports, and demonstration scripts.
- `train_*.py`: Research scripts used to train and evaluate the OpenAVFF checkpoints.

## Installation & Environment

### Requirements
- Python 3.9+
- Node.js 18+
- FFmpeg (must be installed and accessible in system PATH)
- NVIDIA GPU (RTX 30XX/40XX recommended) with CUDA 11.8+

### 1. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

## Running the Application

MediaDNA operates as a decoupled microservice stack.

**Start the Inference Backend:**
```bash
python run_backend.py
```
*The backend runs on `http://127.0.0.1:8000`. The PyTorch checkpoint is loaded into VRAM on startup.*

**Start the Frontend:**
```bash
cd frontend
npm run dev
```
*The frontend is available at `http://localhost:5173`. If the backend is offline, the UI will safely enter a SYSTEM OFFLINE mode.*

## Demonstration Lab
You can test the exact failure modes of the model without uploading files by navigating to the **Forensic Demo Lab** on the frontend dashboard. It utilizes four local files from the `FakeAVCeleb_v1.2` dataset to demonstrate `REAL/REAL`, `REAL/FAKE`, `FAKE/REAL`, and `FAKE/FAKE` permutations live on your GPU.

## Security & Privacy Note
- **Local Only**: This system is designed to run locally. Uploaded media is stored in `uploads/` temporarily and deleted immediately after inference via a `finally` block constraint.
- **History**: Inference results (metadata, logits, scores) are archived in `backend/history.json` for deep-linking. Video files themselves are not retained.

## License
MIT License. See `LICENSE` for details.