import os
import torch
import hashlib

ckpt_path = 'checkpoints/v14_fullscale/models/best_audio_model.pth'
output_file = 'V22_2_CHECKPOINT_PROVENANCE.md'

if not os.path.exists(ckpt_path):
    print("Checkpoint not found!")
else:
    size = os.path.getsize(ckpt_path)
    
    sha256 = hashlib.sha256()
    with open(ckpt_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    sha256_hex = sha256.hexdigest()
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    param_count = sum(p.numel() for p in ckpt.values())
    
    content = f"""# V22.2 Checkpoint Provenance

## File Information
- **Absolute Path:** `{os.path.abspath(ckpt_path)}`
- **SHA256:** `{sha256_hex}`
- **Size:** `{size} bytes`
- **Parameter Count:** `{param_count}`
- **Architecture:** `UNKNOWN` (Only state_dict keys are present, likely a multimodal VideoCAVMAEFT variant)

## Provenance Evidence
- **Training Dataset:** `UNKNOWN` (No metadata embedded)
- **Training Manifest:** `UNKNOWN`
- **Validation Set:** `UNKNOWN`
- **Training Configuration:** `UNKNOWN`
- **Epoch:** `UNKNOWN`
- **Optimizer:** `UNKNOWN`
- **Seed:** `UNKNOWN`

## Conclusion
The provenance of the active V14 checkpoint is `UNKNOWN`. It lacks structural metadata, epoch history, and manifest tracking.
"""
    with open(output_file, 'w') as f:
        f.write(content)
    print("Created V22_2_CHECKPOINT_PROVENANCE.md")
