# Checkpoint Forensics Audit

## Active checkpoint: PARTIAL provenance

| Field | Evidence |
|---|---|
| Path | `checkpoints/v14_fullscale/models/best_audio_model.pth` |
| Size | 748,199,397 bytes |
| SHA256 | `108dccfc4a29ab0a59fb31b60c11aa928b08641c2a1c58db0239d8027c5531a9` |
| Load structure | `OrderedDict`, 313 tensors |
| Tensor parameter count | 187,015,458 float32 values |
| Key namespace | `module.audio_encoder`, visual/fusion keys; DataParallel-formatted |
| Runtime loader | `VideoCAVMAEFT`, then `DataParallel`, `load_state_dict(..., strict=False)` |
| Training config | V14 text config names `train_v14.csv`, `val_v14.csv`, base LR 1e-5, head multiplier 50 |
| Training dataset/provenance | UNKNOWN: named manifests are absent |
| Training epoch | PARTIAL: five result rows exist, checkpoint-to-row association is not recorded |
| Optimizer | PARTIAL: `best_optim_state.pth` has `state` and `param_groups`, but no checkpoint binding/epoch is recorded |

Additional checkpoint families (v8 and v16.1) are present, but no active API configuration points to them. Their files were not accepted as independent candidates because their split provenance is likewise unavailable.

`strict=False` means a successful load does not prove that all expected weights were loaded. A reproducible candidate needs the model constructor version, full state-dict compatibility report, manifest hashes, seed, software versions, epoch, optimizer association, and training command in one immutable run record.
