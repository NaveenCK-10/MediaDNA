# MediaDNA Deployment Guide (Final)

## Prerequisites
- Python 3.9+
- Node.js 16+
- CUDA-compatible GPU (min 8GB VRAM)

## Setup Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload
```

## Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

## Production Considerations
- **Operating-Point Tuning**: The current DEV-derived operating-point reference is tuned for the development distribution. Production deployments require site-specific calibration.
- **Model Size**: AVFF requires substantial VRAM; process jobs sequentially.
