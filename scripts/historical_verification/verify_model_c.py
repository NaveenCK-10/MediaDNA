import torch
import hashlib
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

def verify_checkpoint(ckpt_path, sha_path):
    # 1. Verify SHA-256
    with open(sha_path, "r") as f:
        expected_sha = f.read().strip()

    sha256 = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    actual_sha = sha256.hexdigest()

    if expected_sha != actual_sha:
        print(f"FAIL: SHA-256 mismatch. Expected {expected_sha}, got {actual_sha}")
        sys.exit(1)
    print(f"SHA-256 matched: {actual_sha}")

    # 2. Verify Checkpoint reload in fresh process
    from src.models.audio_specialist import AudioSpecialist
    model = AudioSpecialist()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Quick forward pass
    dummy = torch.randn(1, 1024, 128)
    with torch.no_grad():
        out = model(dummy)
    print(f"Checkpoint reload OK. Output shape: {out.shape}")

    # 3. Update pipeline state
    state_file = os.path.join(PROJECT_ROOT, "MEDIADNA_PIPELINE_STATE.json")
    with open(state_file, "r") as f:
        state = json.load(f)
    
    state["phases"]["PHASE_5_MODEL_C"]["status"] = "COMPLETED"
    state["phases"]["PHASE_5_MODEL_C"]["checkpoint"] = "V22_3C_AUDIO_CHECKPOINT.pth"
    state["phases"]["PHASE_5_MODEL_C"]["results"] = "V22_3C_AUDIO_REPAIRED_REPORT.md"
    state["phases"]["PHASE_5_MODEL_C"]["timestamp"] = datetime.now().isoformat()
    state["phases"]["PHASE_5_MODEL_C"]["hash"] = actual_sha

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Updated pipeline state: PHASE_5_MODEL_C = COMPLETED")

if __name__ == "__main__":
    verify_checkpoint(
        os.path.join(PROJECT_ROOT, "V22_3C_AUDIO_CHECKPOINT.pth"),
        os.path.join(PROJECT_ROOT, "V22_3C_AUDIO_CHECKPOINT.sha256")
    )
