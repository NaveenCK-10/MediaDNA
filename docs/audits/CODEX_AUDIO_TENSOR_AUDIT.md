# CODEX Audio Tensor Contract Audit

**Verdict: CONFIRMED.** The `MelSpectrogram` path must transpose its
frequency-major result before calling `VideoCAVMAEFT`. The normal current V22.2
baseline path does this and is technically correct. This verdict is based on
source and a deterministic runtime trace, not on the AUC change.

**Does the current V22.2 audio preprocessing correctly feed `AudioEncoder`?**
**YES**, for its normal, successfully decoded-audio path. Its missing-audio
fallback at `tools/v22_2_baseline_eval.py:50-51` is an exception: it constructs
`(1, 128, 1024)`, which is not the contract.

**Is the reported AUC 0.8974 technically attributable to this correction?**
**CANNOT DETERMINE.** The current result is reproducible from its prediction
file, and the tensor correction is independently valid; however, the repository
does not preserve a paired before/after prediction run with identical checkpoint,
manifest, and all other settings. Checkpoint training provenance is also marked
unknown. The metric is therefore supporting evidence, not a causal proof.

## Actual contract and full path

The following is the actual normal V22.2 baseline path, with dimensions and
their meanings:

| Stage | Tensor shape | Meaning / source evidence |
|---|---:|---|
| Mono, 16 kHz waveform | `(samples,)` | `v22_2_baseline_eval.py:53-58` extracts mono audio and applies `MelSpectrogram`. |
| Native `MelSpectrogram` | `(128, raw_frames)` | `torchaudio.transforms.MelSpectrogram(..., n_mels=128)` produces frequency-major output. Runtime produced `(128, 1025)` for the deterministic 10.240 s signal. |
| Log/normalization/crop-or-pad | `(128, 1024)` | Lines 60-68 transform only values and the trailing time dimension. Here dim 0 is frequency and dim 1 is time. |
| **V22.2 baseline model input** | **`(B, 1024, 128)`** | Lines 70-71 transpose dimensions 0 and 1, then add batch. The dimensions are batch, time frames, mel-frequency bins. |
| `AudioEncoder` public input | `(B, T=1024, F=128)` | The independently derived mapping is confirmed by the operations at `audio_modules.py:80-81`; it is not inferred from variable names. |
| Encoder Conv2d input | `(B, 1, F=128, T=1024)` | `unsqueeze(1)` creates channel, then `permute(0,1,3,2)` moves original dim 2 to height and original dim 1 to width. |
| Conv patch grid | `(B, 768, 8, 64)` | 16×16 Conv2d kernel/stride over 128×1024 creates 8 frequency patches by 64 time patches. |
| Patch tokens | `(B, 512, 768)` | The custom `PatchEmbed` applies `Conv2d(...).flatten(2).transpose(1,2)`, so 8×64 = 512 tokens. |
| Transformer output | `(B, 512, 768)` | Position/modality embeddings are added at `audio_modules.py:86-91`; all 12 transformer blocks retain this shape. |
| Fine-tuning fusion/classifier | audio tokens reshape to `(B, 8, 64, 768)` | `video_cav_mae.py:367-398` calls `audio_encoder(audio)`, reshapes 512 tokens into 8 groups, fuses with visual features, reduces to 512 audio scalars, then `mlp_audio(512)` and `mlp_head` produce `(B, 2)`. |

The public encoder contract is therefore **rank 3, `(B, 1024, 128)`**, not
`(B, 1, 128, 1024)`. The latter is an internal Conv2d representation created by
the encoder itself.

## Patching and semantic axes

`PatchEmbed` creates `nn.Conv2d(in_chans=1, embed_dim=768,
kernel_size=(16,16), stride=(16,16))` in `audio_modules.py:15-26`; the encoder
instantiates it at line 53. The actual indexing proves the axes:

```text
public input                 (B, time, frequency)
unsqueeze(1)                 (B, channel, time, frequency)
permute(0, 1, 3, 2)          (B, channel, frequency, time)
Conv2d output                (B, 768, frequency_patches, time_patches)
flatten(2).transpose(1,2)    (B, frequency_patches*time_patches, 768)
```

Thus patch height is frequency, patch width is time, both patch size and stride
are 16, and the intended grid is **8 frequency × 64 time**. The position table
is initialized with an 8-by-64 layout (`audio_modules.py:96`), consistent with
that grid. A wrong public input reverses the Conv grid to 64-by-8 while retaining
the same total of 512 tokens; that is why it executes without a shape exception.

## Runtime experiment

`tools/codex_audio_shape_test.py` is the requested read-only diagnostic. It
uses no media files, checkpoints, or production writes. It generates a fixed
440 Hz tone plus a 3.2 kHz burst from 6.0 to 7.0 seconds and records hooks on
the patch Conv2d, `PatchEmbed`, and first transformer block. Its output is in
`CODEX_AUDIO_SHAPE_DIAGNOSTIC.txt`.

Observed with torch/torchaudio 2.6.0 on CPU:

| Tested audio supplied to `AudioEncoder` | Patch Conv2d input | Patch Conv2d output | Encoder result |
|---|---:|---:|---|
| Native Mel `(1, 128, 1025)` | `(1, 1, 1025, 128)` | `(1, 768, 64, 8)` | passes, but semantically inverted |
| Current V22.2 `(1, 1024, 128)` | `(1, 1, 128, 1024)` | `(1, 768, 8, 64)` | passes and matches contract |
| Untransposed `(1, 128, 1024)` | `(1, 1, 1024, 128)` | `(1, 768, 64, 8)` | passes, but semantically inverted |

All successful runs yielded `(1, 512, 768)` patch tokens and transformer output.
That equality is precisely why a runtime exception is not a useful discriminator.
The script also executes the complete `VideoCAVMAEFT(audio, video)` fusion and
classifier once using the current `(1,1024,128)` audio input and a deterministic
zero video tensor `(1,3,16,224,224)`; its output shape is `(1,2)`.

The semantic test independently located the persistent 440 Hz tone at mel bin
24. It located the extra 3.2 kHz burst at mel bin 87 and time frame 633, within
the intended 600-700-frame window. With an untransposed input, encoder input dim
0 is interpreted as time, so index 87 designates that *frequency* row instead
of a time frame. The observed 64×8 patch geometry consequently has frequency
and time exchanged.

## Forward argument order and consumers

The model definition is `VideoCAVMAEFT.forward(self, audio, video)` at
`src/models/video_cav_mae.py:367`. It immediately uses
`self.audio_encoder(audio)` then `self.visual_encoder(video)` (lines 371-373).
Production inference calls `self.model(a_input, v_input)` at
`backend/inference.py:194-198`; V22.1 training/evaluation calls
`model(a_input, v_input)` at `V22_1_EXECUTION.py:99-118`; V22.2 evaluation calls
`self.model(a_t, v_t)` at `tools/v22_2_baseline_eval.py:102-104`. No reversed
production/evaluation call site was found in the inspected paths.

The separate pretraining `VideoCAVMAE.forward(audio, video, ...)` has the same
argument order at `src/models/video_cav_mae.py:220`. It passes the same audio
encoder and, during reconstruction, explicitly transposes `(B,T,F)` to
`(B,1,F,T)` at lines 252-254, independently corroborating the contract.

## Before/after evidence

The repository has no committed V22.2 baseline history: the V22.2 files are
untracked in the current checkout, so an exact Git revision-to-revision diff is
not available. The available source nevertheless contains a direct untransposed
V22.2 example and the corrected evaluation operation:

| File / line | Old operation | New operation | Tensor effect |
|---|---|---|---|
| `tools/v22_2_sanity_test.py:49-59` | `a_tensor = mel_spec.unsqueeze(0)` followed by time crop/pad | — | Sends `(B,F,T)` to the model; encoder turns it into `(B,1,T,F)`, yielding a 64×8 grid. |
| `tools/v22_2_baseline_eval.py:63-71` | Same pre-transpose state is `(F,T)` | `mel_spec = mel_spec.transpose(0,1)` then `unsqueeze(0)` | Sends `(B,T,F)`; encoder produces `(B,1,F,T)`, yielding the trained 8×64 grid. |
| `tools/v22_2_baseline_eval.py:50-51` | Missing-audio fallback `zeros(1,128,1024)` | no corrective transpose | This exceptional fallback remains wrong under the verified contract. |

Therefore the transpose is a real correction; the statement that it prevents
time/frequency-inverted patching is technically accurate. The claim that it
alone caused a particular metric change is not established by the available
artifacts.

## Historical preprocessing check

| Version | Evidence | Tensor at model boundary | Finding |
|---|---|---|---|
| V14 | `src/dataloader.py:120-142, 329-335`; V14 checkpoint is the active baseline checkpoint | Kaldi fbank `(T,128)`, batched by DataLoader | Time-major; no observed mismatch. |
| V15 | `v15_2_experiments.py` uses `VideoAudioEvalDataset`, which returns the same `src/dataloader.py` fbank | `(B,T,128)` | Time-major; no observed mismatch. |
| V21 | No independently executable V21 preprocessing implementation is retained; `v21_master_eval.py` is a report-generation stub. The shared backend path is time-major. | Not independently reconstructible from a V21 run artifact | **UNKNOWN** for V21-specific provenance; no source evidence of an inverted boundary was found. |
| V21.4 | `V21_4_BENCHMARK.py:58-60,75-80` | `(B,1024,128)` | Time-major; no observed mismatch. |
| V22.1 | `V22_1_EXECUTION.py:32-40,99-118` | `(B,1024,128)` | Time-major; no observed mismatch. |
| V22.2 | `tools/v22_2_sanity_test.py` lacks transpose; baseline evaluator has it | sanity path `(B,128,1024)`; baseline normal path `(B,1024,128)` | A V22.2 `MelSpectrogram`-specific / experiment-path issue, not a demonstrated historical dataloader bug. |

The supported classification is therefore: **new V22.2 experiment-specific
bug**, with the caveat that the missing pre-change baseline revision prevents a
stronger claim about which exact prior run used it.

## V22.2 DEV result verification

I independently read `V22_2_BASELINE_PREDICTIONS.csv` (2,264 rows) and computed
the metrics without rewriting any benchmark artifact:

| Measure at threshold 0.60 | Recomputed value |
|---|---:|
| Threshold-free ROC AUC | 0.8973803071 |
| Balanced accuracy | 0.8633694670 |
| F1 | 0.8417473189 |
| MCC | 0.2355339774 |
| Confusion matrix `[ [TN, FP], [FN, TP] ]` | `[[50, 0], [605, 1609]]` |

Category counts read from the same prediction artifact are `RVRA=50`,
`RVFA=50`, `FVRA=1057`, and `FVFA=1107`; they match
`V22_2_CLASS_DISTRIBUTION.json`'s DEV split.

The selection code shows that threshold 0.60 was selected by maximizing MCC over
the fixed sweep in `tools/v22_2_baseline_eval.py:124-155`. Its entry point uses
only `data/v22_2_dev.csv` (lines 270-274). This evaluator neither opens the CAL
nor locked TEST manifest, so the available source supports DEV-only selection.
No new threshold was optimized in this audit.

## Limitations

This audit does not infer checkpoint-training provenance, reconstruct a missing
V22.2 Git history, or use an AUC difference as proof of the tensor contract.
It establishes the contract from the actual permutation, Conv2d geometry,
positional grid, deterministic signal behavior, and runtime hooks.
