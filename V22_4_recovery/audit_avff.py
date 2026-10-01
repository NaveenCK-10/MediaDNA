import os
import sys
import torch
import hashlib
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.models.video_cav_mae import VideoCAVMAEFT

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def audit_checkpoint():
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found!")
        return

    sha256 = get_sha256(ckpt_path)
    
    # Load model
    model = VideoCAVMAEFT()
    model_state = model.state_dict()
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if 'model' in ckpt:
        ckpt_state = ckpt['model']
    elif 'state_dict' in ckpt:
        ckpt_state = ckpt['state_dict']
    else:
        ckpt_state = ckpt

    # DataParallel unwrapping if needed
    if list(ckpt_state.keys())[0].startswith('module.'):
        ckpt_state = {k[7:]: v for k, v in ckpt_state.items()}

    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())
    
    matched_keys = model_keys.intersection(ckpt_keys)
    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys
    
    # Check shapes for matched
    shape_mismatch = []
    for k in matched_keys:
        if model_state[k].shape != ckpt_state[k].shape:
            shape_mismatch.append((k, model_state[k].shape, ckpt_state[k].shape))

    audit = {
        "checkpoint_path": ckpt_path,
        "sha256": sha256,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_tensor_count": len(ckpt_state),
        "model_tensor_count": len(model_state),
        "matched_keys_count": len(matched_keys),
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "shape_mismatches_count": len(shape_mismatch),
        "missing_keys_sample": list(missing_keys)[:10],
        "unexpected_keys_sample": list(unexpected_keys)[:10],
        "shape_mismatches": shape_mismatch
    }
    
    with open("AVFF_AUDIT.json", "w") as f:
        json.dump(audit, f, indent=2)

if __name__ == "__main__":
    audit_checkpoint()
