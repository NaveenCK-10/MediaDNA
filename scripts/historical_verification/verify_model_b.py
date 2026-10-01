import torch
import hashlib
import json
import os
import sys

def verify_checkpoint(ckpt_path, sha_path):
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        sys.exit(1)
        
    if not os.path.exists(sha_path):
        print(f"Error: {sha_path} not found.")
        sys.exit(1)

    # 1. Verify SHA-256
    with open(sha_path, "r") as f:
        expected_sha = f.read().strip()

    sha256 = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    actual_sha = sha256.hexdigest()

    if expected_sha != actual_sha:
        print(f"Error: SHA-256 mismatch. Expected {expected_sha}, got {actual_sha}")
        sys.exit(1)
    
    print(f"SHA-256 matched: {actual_sha}")

    # 2. Verify Checkpoint reload
    try:
        # Load weights only for simplicity, assuming simple state dict
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("Successfully reloaded checkpoint in fresh process.")
    except Exception as e:
        print(f"Error reloading checkpoint: {e}")
        sys.exit(1)

    # 3. Update pipeline state
    state_file = "MEDIADNA_PIPELINE_STATE.json"
    with open(state_file, "r") as f:
        state = json.load(f)
    
    state["phases"]["PHASE_4_MODEL_B"]["status"] = "COMPLETED"
    state["phases"]["PHASE_4_MODEL_B"]["checkpoint"] = ckpt_path
    state["phases"]["PHASE_4_MODEL_B"]["results"] = "V22_3B_VISUAL_REPAIRED_REPORT.md"
    state["phases"]["PHASE_4_MODEL_B"]["hash"] = actual_sha

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
        
    print(f"Updated {state_file} with PHASE_4_MODEL_B completion.")

if __name__ == "__main__":
    verify_checkpoint("V22_3B_VISUAL_CHECKPOINT.pth", "V22_3B_VISUAL_CHECKPOINT.sha256")
