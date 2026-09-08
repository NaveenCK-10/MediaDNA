# MediaDNA

MediaDNA is a robust inference wrapper and user interface built around the OpenAVFF deep learning architecture. It enhances standard model inference with deterministic visual anomaly tracking for comprehensive synthetic media detection.

## Features

*   **OpenAVFF Engine**: Runs the frozen `best_audio_model.pth` VideoCAVMAEFT model with fixed deterministic preprocessing.
*   **Visual Consistency**: Deterministic measurement of blur variance and inter-frame volatility to expose splicing and frame-by-frame deepfake generation artifacts.
*   **Experimental Fusion**: Combines OpenAVFF probability with visual anomalies to assist in finding "FakeVideo-RealAudio" deepfakes.
*   **Modern Web GUI**: A React + FastAPI dashboard to easily upload and analyze videos locally.

## Setup Instructions

Ensure you have Anaconda installed with the `avff` environment active.

```cmd
conda activate avff
```

### 1. Start the Backend (FastAPI)

Open a new command prompt:

```cmd
cd C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF
conda activate avff
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend (React/Vite)

Open another command prompt:

```cmd
cd C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\frontend
npm run dev
```

The website will be available at [http://localhost:5173](http://localhost:5173).

## Evaluation

To reproduce the held-out evaluation and ablation study:

```cmd
conda activate avff
python evaluate_mediadna.py
```

This will generate evaluation metrics in the `reports/` folder.

## CLI vs GUI Consistency

To verify that the GUI backend matches the CLI OpenAVFF engine exactly (tolerance <= 0.001):

```cmd
conda activate avff
python compare_cli_gui.py
```
