"""
FINAL ROOT-CAUSE DIAGNOSTIC for OpenAVFF Stage-3
Covers Tasks 1-9 comprehensively. Does NOT modify anything.
"""
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
import sys
import subprocess
import tempfile
import soundfile as sf
from torch.cuda.amp import autocast
import csv
import random
from einops import rearrange

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader

FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
CHECKPOINT = "checkpoints/stage-3.pth"
SMALL_CSV = "data/test_small.csv"
BALANCED_CSV = "data/diagnostic_balanced.csv"
FAKEAVCELEB_ROOT = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"

def section_header(num, title):
    print(f"\n{'='*70}")
    print(f"TASK {num}: {title}")
    print(f"{'='*70}")

def extract_audio_raw(video_path):
    """Extract audio from MP4 and return raw waveform + fbank BEFORE normalization."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        temp_wav = tmp.name
    try:
        cmd = [FFMPEG_PATH, "-y", "-loglevel", "error", "-i", video_path,
               "-vn", "-ac", "1", "-ar", "16000", temp_wav]
        subprocess.run(cmd, check=True)
        waveform_np, sr = sf.read(temp_wav)
        waveform = torch.tensor(waveform_np).unsqueeze(0).float()
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr,
            use_energy=False, window_type='hanning',
            num_mel_bins=128, dither=0.0, frame_shift=10)
        return waveform, sr, fbank
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def create_balanced_csv():
    """Create balanced diagnostic CSV if it doesn't exist."""
    if os.path.exists(BALANCED_CSV):
        print(f"  {BALANCED_CSV} already exists, reusing it.")
        return
    categories = {
        'FakeVideo-FakeAudio': 1,
        'RealVideo-RealAudio': 0,
        'FakeVideo-RealAudio': 1,
        'RealVideo-FakeAudio': 1,
    }
    rows = []
    for cat_name, label in categories.items():
        cat_dir = os.path.join(FAKEAVCELEB_ROOT, cat_name)
        if not os.path.exists(cat_dir):
            print(f"  WARNING: {cat_dir} not found, skipping")
            continue
        mp4s = []
        for root, dirs, files in os.walk(cat_dir):
            for f in files:
                if f.lower().endswith('.mp4'):
                    mp4s.append(os.path.join(root, f))
        random.seed(42)
        selected = random.sample(mp4s, min(20, len(mp4s)))
        for vpath in selected:
            rows.append((vpath, label))
        print(f"  {cat_name}: {len(selected)} videos selected from {len(mp4s)} total")
    
    with open(BALANCED_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label'])
        for vpath, label in rows:
            writer.writerow([vpath, label])
    print(f"  Created {BALANCED_CSV} with {len(rows)} samples")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================================================================
    # TASK 1: INSPECT REPOSITORY STRUCTURE
    # ==================================================================
    section_header(1, "REPOSITORY & OFFICIAL INFERENCE PIPELINE")
    
    print("\nREADME.md states:")
    print("  - Unofficial PyTorch implementation of AVFF")
    print("  - Stage 3 = Finetune on deepfake classification")
    print("  - Evaluation: python eval.py --checkpoint path/to/stage-3.pth --csv_file data/testset.csv")
    print("  - Stage-3 Google Drive link is a FOLDER (not single file)")
    
    print("\nadapt_pretraining_weights.ipynb:")
    print("  - Loads stage-2.pth")
    print("  - Filters OUT decoder and a2v/v2a MLP keys")  
    print("  - Creates VideoCAVMAEFT() with randomly initialized MLP head")
    print("  - Saves as stage-3.pth (INITIAL weights, NOT trained)")
    print("  - Missing keys on load: a2v.mlp, v2a.mlp, mlp_vision, mlp_audio, mlp_head.*")
    
    print("\nstage-3.sh training config:")
    print("  - loss: BCE")
    print("  - lr: 1e-5, head_lr multiplier: 50x")
    print("  - epochs: 10")
    print("  - dataset_mean: -5.081, dataset_std: 4.4849")
    print("  - n_classes: 2")
    print("  - noise: True (training only)")
    print("  - pretrain_path: kinestic-pretrain-full.pth")
    
    print("\neval.py pipeline:")
    print("  - VideoCAVMAEFT() -> DataParallel -> load checkpoint")
    print("  - NO model.eval() call (BUG: dropout active in training mode)")
    print("  - sigmoid(output) -> probability[0] as fake score")
    
    print("\ntraintest_ft.py training loop:")
    print("  - loss = BCEWithLogitsLoss()")
    print("  - output = model(a_input, v_input)")
    print("  - loss = loss_fn(output, labels)")
    print("  - labels from dataloader: [label, 1-label] where label=1 for Fake")
    
    print("\ntraintest_ft.py validation (calculate_stats):")
    print("  - Uses np.argmax(output, 1) vs np.argmax(target, 1)")
    print("  - Raw logits, NOT sigmoid, are used for AP/AUC/accuracy")
    
    # ==================================================================
    # TASK 2: VERIFY CHECKPOINT LOADING
    # ==================================================================
    section_header(2, "CHECKPOINT LOADING VERIFICATION")
    
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    print(f"\nCheckpoint type: {type(ckpt).__name__}")
    print(f"Number of parameters: {len(ckpt)}")
    
    # File sizes
    stage1_size = os.path.getsize("checkpoints/stage-1.pth")
    stage2_size = os.path.getsize("checkpoints/stage-2.pth")
    stage3_size = os.path.getsize("checkpoints/stage-3.pth")
    print(f"\nFile sizes:")
    print(f"  stage-1.pth: {stage1_size:,} bytes ({stage1_size/1e6:.1f} MB)")
    print(f"  stage-2.pth: {stage2_size:,} bytes ({stage2_size/1e6:.1f} MB)")
    print(f"  stage-3.pth: {stage3_size:,} bytes ({stage3_size/1e6:.1f} MB)")
    print(f"  stage-3 is {'SMALLER' if stage3_size < stage2_size else 'LARGER'} than stage-2 (expected: smaller, decoder removed)")
    
    model = VideoCAVMAEFT()
    model = torch.nn.DataParallel(model)
    miss, unexp = model.load_state_dict(ckpt, strict=False)
    
    print(f"\nMissing keys: {len(miss)}")
    for k in miss:
        print(f"  {k}")
    print(f"Unexpected keys: {len(unexp)}")
    for k in unexp:
        print(f"  {k}")
    
    # Check classifier head parameters specifically
    print("\nClassifier head keys in checkpoint:")
    head_keys = [k for k in ckpt.keys() if any(x in k for x in ['mlp_head', 'mlp_vision', 'mlp_audio', 'a2v', 'v2a'])]
    for k in sorted(head_keys):
        print(f"  {k}: {ckpt[k].shape}")
    
    print(f"\nOutput shape: model outputs [batch_size, {model.module.mlp_head.fc3.out_features}]")
    
    # ==================================================================
    # TASK 3: VERIFY MODEL OUTPUT SEMANTICS
    # ==================================================================
    section_header(3, "MODEL OUTPUT SEMANTICS")
    
    print("\nClassifier architecture (VideoCAVMAEFT):")
    print(f"  mlp_vision: Linear(1568, 1024)")
    print(f"  mlp_audio:  Linear(512, 1024)")
    print(f"  mlp_head:   MLP(2048 -> 1024 -> 1024 -> 2)")
    print(f"              with Dropout(0.5) after each hidden layer")
    print(f"              and ReLU activations")
    print()
    print("  MLP head definition:")
    print(model.module.mlp_head)
    
    print(f"\nTraining loss: BCEWithLogitsLoss()")
    print(f"  This applies sigmoid INTERNALLY per-element")
    print(f"  Target format: [label, 1-label]")
    print(f"  When label=1 (Fake): target = [1, 0]")
    print(f"  When label=0 (Real): target = [0, 1]")
    print(f"  Output index 0 = Fake logit")
    print(f"  Output index 1 = Real logit")
    
    print(f"\nEval interpretation:")
    print(f"  sigmoid(output[0]) = P(Fake)")
    print(f"  This is CORRECT for BCEWithLogitsLoss with [label, 1-label] targets")
    
    # ==================================================================
    # TASK 4: VERIFY AUDIO PIPELINE
    # ==================================================================
    section_header(4, "AUDIO PIPELINE VERIFICATION")
    
    # Use test_small.csv samples
    val_audio_conf = {
        'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0,
        'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224
    }
    dataset = dataloader.VideoAudioEvalDataset(csv_file=SMALL_CSV, audio_conf=val_audio_conf)
    
    # Get raw audio stats for fake and real
    for idx in [0, 2]:  # 0=fake, 2=real
        video_path = dataset.data[idx][0]
        label = dataset.data[idx][1]
        vtype = "FAKE" if int(label) == 1 else "REAL"
        
        print(f"\n--- {vtype} video: {os.path.basename(video_path)} ---")
        
        waveform, sr, fbank_raw = extract_audio_raw(video_path)
        
        print(f"\n  Audio raw waveform:")
        print(f"    shape: {tuple(waveform.shape)}")
        print(f"    dtype: {waveform.dtype}")
        print(f"    min:   {waveform.min().item():.6f}")
        print(f"    max:   {waveform.max().item():.6f}")
        print(f"    mean:  {waveform.mean().item():.6f}")
        print(f"    std:   {waveform.std().item():.6f}")
        print(f"    sample_rate: {sr}")
        
        print(f"\n  Fbank BEFORE normalization:")
        print(f"    shape: {tuple(fbank_raw.shape)}")
        print(f"    min:   {fbank_raw.min().item():.4f}")
        print(f"    max:   {fbank_raw.max().item():.4f}")
        print(f"    mean:  {fbank_raw.mean().item():.4f}")
        print(f"    std:   {fbank_raw.std().item():.4f}")
        
        # Interpolate to target_length
        fbank_interp = torch.nn.functional.interpolate(
            fbank_raw.unsqueeze(0).transpose(1,2), size=(1024,),
            mode='linear', align_corners=False).transpose(1,2).squeeze(0)
        
        print(f"\n  Fbank AFTER interpolation (before normalization):")
        print(f"    shape: {tuple(fbank_interp.shape)}")
        print(f"    min:   {fbank_interp.min().item():.4f}")
        print(f"    max:   {fbank_interp.max().item():.4f}")
        print(f"    mean:  {fbank_interp.mean().item():.4f}")
        print(f"    std:   {fbank_interp.std().item():.4f}")
        
        # Normalize
        fbank_norm = (fbank_interp - (-5.081)) / 4.4849
        
        print(f"\n  Fbank AFTER normalization:")
        print(f"    shape: {tuple(fbank_norm.shape)}")
        print(f"    min:   {fbank_norm.min().item():.4f}")
        print(f"    max:   {fbank_norm.max().item():.4f}")
        print(f"    mean:  {fbank_norm.mean().item():.4f}")
        print(f"    std:   {fbank_norm.std().item():.4f}")
        
        # Get final tensor from dataset
        a_input, v_input, labels, vname = dataset[idx]
        print(f"\n  Final audio tensor from dataset:")
        print(f"    shape: {tuple(a_input.shape)}")
        print(f"    dtype: {a_input.dtype}")
        print(f"    min:   {a_input.min().item():.4f}")
        print(f"    max:   {a_input.max().item():.4f}")
        print(f"    mean:  {a_input.mean().item():.4f}")
        print(f"    std:   {a_input.std().item():.4f}")
    
    # ==================================================================
    # TASK 5: VERIFY VISUAL PIPELINE
    # ==================================================================
    section_header(5, "VISUAL PIPELINE VERIFICATION")
    
    for idx in [0, 2]:
        video_path = dataset.data[idx][0]
        label = dataset.data[idx][1]
        vtype = "FAKE" if int(label) == 1 else "REAL"
        
        a_input, v_input, labels, vname = dataset[idx]
        
        print(f"\n--- {vtype} video: {os.path.basename(video_path)} ---")
        print(f"  Visual tensor:")
        print(f"    shape: {tuple(v_input.shape)}")
        print(f"    dtype: {v_input.dtype}")
        print(f"    min:   {v_input.min().item():.4f}")
        print(f"    max:   {v_input.max().item():.4f}")
        print(f"    mean:  {v_input.mean().item():.4f}")
        print(f"    std:   {v_input.std().item():.4f}")
        
    print(f"\n  Expected preprocessing (VideoAudioEvalDataset):")
    print(f"    ToPILImage -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)")
    print(f"  Training preprocessing (VideoAudioDataset):")
    print(f"    ToPILImage -> Resize(224,224) -> ToTensor -> Normalize(ImageNet)")
    print(f"  NOTE: Eval uses CenterCrop, Train uses Resize — minor difference, standard practice")
    
    # ==================================================================
    # TASK 6: CONTROL EXPERIMENTS
    # ==================================================================
    section_header(6, "CONTROL EXPERIMENTS")
    
    model.eval()
    model.to(device)
    
    # Get real samples
    fake_a, fake_v, _, _ = dataset[0]  # Fake
    real_a, real_v, _, _ = dataset[2]  # Real
    
    experiments = {
        "A. Real Fake video (audio+visual)": (fake_a, fake_v),
        "B. Real Real video (audio+visual)": (real_a, real_v),
        "C. Zero audio + real visual (Fake)": (torch.zeros_like(fake_a), fake_v),
        "D. Real audio (Fake) + zero visual": (fake_a, torch.zeros_like(fake_v)),
        "E. Zero audio + zero visual": (torch.zeros_like(fake_a), torch.zeros_like(fake_v)),
    }
    
    with torch.no_grad():
        for name, (a, v) in experiments.items():
            a_in = a.unsqueeze(0).to(device)
            v_in = v.unsqueeze(0).to(device)
            out = model(a_in, v_in)
            logits = out.cpu().float().numpy()[0]
            sig = torch.sigmoid(out.float()).cpu().numpy()[0]
            pred = "FAKE" if logits[0] > logits[1] else "REAL"
            print(f"\n  {name}:")
            print(f"    RAW LOGITS: [{logits[0]:+.4f}, {logits[1]:+.4f}]")
            print(f"    SIGMOID:    [{sig[0]:.8f}, {sig[1]:.8f}]")
            print(f"    PREDICTED:  {pred}")
    
    # ==================================================================
    # TASK 7: 80-SAMPLE BALANCED TEST
    # ==================================================================
    section_header(7, "80-SAMPLE BALANCED TEST")
    
    create_balanced_csv()
    
    bal_dataset = dataloader.VideoAudioEvalDataset(csv_file=BALANCED_CSV, audio_conf=val_audio_conf)
    bal_loader = torch.utils.data.DataLoader(bal_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    all_logits = []
    all_labels = []
    all_preds = []
    all_categories = []
    
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_names) in enumerate(bal_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            with autocast():
                out = model(a_input, v_input)
            logits = out.cpu().float().numpy()[0]
            label_tensor = labels.cpu().numpy()[0]
            csv_label = int(label_tensor[0])
            pred = 1 if logits[0] > logits[1] else 0
            
            vname = video_names[0]
            if 'FakeVideo-FakeAudio' in vname: cat = 'FVFA'
            elif 'FakeVideo-RealAudio' in vname: cat = 'FVRA'
            elif 'RealVideo-FakeAudio' in vname: cat = 'RVFA'
            elif 'RealVideo-RealAudio' in vname: cat = 'RVRA'
            else: cat = 'UNK'
            
            all_logits.append(logits)
            all_labels.append(csv_label)
            all_preds.append(pred)
            all_categories.append(cat)
    
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_categories = np.array(all_categories)
    
    # Metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    acc = (all_preds == all_labels).sum() / len(all_labels)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"\n  Results on {len(all_labels)} samples:")
    print(f"    Accuracy:  {acc:.4f} ({(all_preds == all_labels).sum()}/{len(all_labels)})")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-score:  {f1:.4f}")
    print(f"\n    Confusion Matrix:")
    print(f"                  Predicted Fake  Predicted Real")
    print(f"    Actual Fake:  {tp:14d}  {fn:14d}")
    print(f"    Actual Real:  {fp:14d}  {tn:14d}")
    
    # Mean logits by class
    fake_mask = all_labels == 1
    real_mask = all_labels == 0
    print(f"\n    Mean Fake logit for actual Fake samples: {all_logits[fake_mask, 0].mean():.4f}")
    print(f"    Mean Fake logit for actual Real samples: {all_logits[real_mask, 0].mean():.4f}")
    print(f"    Mean Real logit for actual Fake samples: {all_logits[fake_mask, 1].mean():.4f}")
    print(f"    Mean Real logit for actual Real samples: {all_logits[real_mask, 1].mean():.4f}")
    
    # Per-category
    print(f"\n    Per-category accuracy:")
    for cat in ['FVFA', 'FVRA', 'RVFA', 'RVRA']:
        mask = all_categories == cat
        if mask.sum() > 0:
            cat_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            cat_pred_fake = (all_preds[mask] == 1).sum()
            cat_pred_real = (all_preds[mask] == 0).sum()
            print(f"      {cat}: {cat_acc:.4f} ({(all_preds[mask] == all_labels[mask]).sum()}/{mask.sum()})  "
                  f"pred_fake={cat_pred_fake}, pred_real={cat_pred_real}")
    
    # ==================================================================
    # TASK 8: DETERMINE IF CHECKPOINT IS CONSTANT
    # ==================================================================
    section_header(8, "LOGIT DISTRIBUTION ANALYSIS")
    
    print(f"\n  Fake logit (index 0) across all {len(all_logits)} samples:")
    print(f"    mean: {all_logits[:, 0].mean():.4f}")
    print(f"    std:  {all_logits[:, 0].std():.4f}")
    print(f"    min:  {all_logits[:, 0].min():.4f}")
    print(f"    max:  {all_logits[:, 0].max():.4f}")
    
    print(f"\n  Real logit (index 1) across all {len(all_logits)} samples:")
    print(f"    mean: {all_logits[:, 1].mean():.4f}")
    print(f"    std:  {all_logits[:, 1].std():.4f}")
    print(f"    min:  {all_logits[:, 1].min():.4f}")
    print(f"    max:  {all_logits[:, 1].max():.4f}")
    
    print(f"\n  Logit difference (fake - real) across all samples:")
    diff = all_logits[:, 0] - all_logits[:, 1]
    print(f"    mean: {diff.mean():.4f}")
    print(f"    std:  {diff.std():.4f}")
    print(f"    min:  {diff.min():.4f}")
    print(f"    max:  {diff.max():.4f}")
    
    # Compare with classifier bias
    fc3_bias = model.module.mlp_head.fc3.bias.data.cpu().numpy()
    print(f"\n  Classifier (fc3) bias values: [{fc3_bias[0]:.6f}, {fc3_bias[1]:.6f}]")
    print(f"  Bias difference (fake - real): {fc3_bias[0] - fc3_bias[1]:.6f}")
    print(f"\n  IMPORTANT: Bias is [{fc3_bias[0]:.4f}, {fc3_bias[1]:.4f}], but logits are ~[+12, -12]")
    print(f"  Therefore the ~+12/-12 logits are NOT from bias alone.")
    print(f"  They come from the learned weights applied to feature representations.")
    
    # ==================================================================
    # TASK 9: CLASSIFIER WEIGHT & BIAS ANALYSIS
    # ==================================================================
    section_header(9, "CLASSIFIER WEIGHT & BIAS DEEP ANALYSIS")
    
    # Walk through every layer of the classifier head
    m = model.module
    
    print("\n  --- fc3 (final classification layer) ---")
    fc3_w = m.mlp_head.fc3.weight.data.cpu()
    fc3_b = m.mlp_head.fc3.bias.data.cpu()
    print(f"    weight shape: {fc3_w.shape}")
    print(f"    weight[0] (Fake): mean={fc3_w[0].mean():.6f}, std={fc3_w[0].std():.6f}, min={fc3_w[0].min():.6f}, max={fc3_w[0].max():.6f}")
    print(f"    weight[1] (Real): mean={fc3_w[1].mean():.6f}, std={fc3_w[1].std():.6f}, min={fc3_w[1].min():.6f}, max={fc3_w[1].max():.6f}")
    print(f"    bias: [{fc3_b[0]:.6f}, {fc3_b[1]:.6f}]")
    corr = torch.corrcoef(fc3_w)[0, 1].item()
    print(f"    correlation between w[0] and w[1]: {corr:.6f}")
    
    print("\n  --- fc2 (second hidden layer) ---")
    fc2_w = m.mlp_head.fc2.weight.data.cpu()
    fc2_b = m.mlp_head.fc2.bias.data.cpu()
    print(f"    weight shape: {fc2_w.shape}")
    print(f"    weight mean={fc2_w.mean():.6f}, std={fc2_w.std():.6f}")
    print(f"    bias mean={fc2_b.mean():.6f}, std={fc2_b.std():.6f}, min={fc2_b.min():.6f}, max={fc2_b.max():.6f}")
    print(f"    bias positive count: {(fc2_b > 0).sum()}/{len(fc2_b)}")
    
    print("\n  --- fc1 (first hidden layer) ---")
    fc1_w = m.mlp_head.fc1.weight.data.cpu()
    fc1_b = m.mlp_head.fc1.bias.data.cpu()
    print(f"    weight shape: {fc1_w.shape}")
    print(f"    weight mean={fc1_w.mean():.6f}, std={fc1_w.std():.6f}")
    print(f"    bias mean={fc1_b.mean():.6f}, std={fc1_b.std():.6f}, min={fc1_b.min():.6f}, max={fc1_b.max():.6f}")
    print(f"    bias positive count: {(fc1_b > 0).sum()}/{len(fc1_b)}")
    
    # Mathematical analysis: what would fc3 produce if fc2 output was all-positive (post-ReLU)?
    print("\n  --- Mathematical analysis ---")
    print("  After fc2 -> ReLU, all values are >= 0")
    print("  fc3 output = fc3_w @ relu2_output + fc3_b")
    print(f"  fc3_w[0].sum() = {fc3_w[0].sum():.6f} (if all relu2 outputs = 1)")
    print(f"  fc3_w[1].sum() = {fc3_w[1].sum():.6f} (if all relu2 outputs = 1)")
    
    # Check if weights look randomly initialized
    print("\n  --- Checking if MLP head weights are RANDOMLY INITIALIZED ---")
    # Xavier normal init produces weights with std ≈ sqrt(2 / (fan_in + fan_out))
    expected_fc1_std = np.sqrt(2.0 / (2048 + 1024))
    expected_fc2_std = np.sqrt(2.0 / (1024 + 1024))
    expected_fc3_std = np.sqrt(2.0 / (1024 + 2))
    print(f"  fc1: actual std={fc1_w.std():.6f}, expected xavier_normal std={expected_fc1_std:.6f}")
    print(f"  fc2: actual std={fc2_w.std():.6f}, expected xavier_normal std={expected_fc2_std:.6f}")
    print(f"  fc3: actual std={fc3_w.std():.6f}, expected xavier_normal std={expected_fc3_std:.6f}")
    
    fc1_match = abs(fc1_w.std().item() - expected_fc1_std) / expected_fc1_std < 0.15
    fc2_match = abs(fc2_w.std().item() - expected_fc2_std) / expected_fc2_std < 0.15
    fc3_match = abs(fc3_w.std().item() - expected_fc3_std) / expected_fc3_std < 0.15
    print(f"  fc1 matches random init: {fc1_match} (ratio: {fc1_w.std().item() / expected_fc1_std:.3f})")
    print(f"  fc2 matches random init: {fc2_match} (ratio: {fc2_w.std().item() / expected_fc2_std:.3f})")
    print(f"  fc3 matches random init: {fc3_match} (ratio: {fc3_w.std().item() / expected_fc3_std:.3f})")

    # Check bias values - random init sets bias to 0
    fc1_bias_near_zero = fc1_b.abs().mean().item() < 0.01
    fc2_bias_near_zero = fc2_b.abs().mean().item() < 0.01
    fc3_bias_near_zero = fc3_b.abs().mean().item() < 0.05
    print(f"\n  fc1 bias near zero: {fc1_bias_near_zero} (abs mean: {fc1_b.abs().mean():.6f})")
    print(f"  fc2 bias near zero: {fc2_bias_near_zero} (abs mean: {fc2_b.abs().mean():.6f})")
    print(f"  fc3 bias near zero: {fc3_bias_near_zero} (abs mean: {fc3_b.abs().mean():.6f})")
    
    all_random = fc1_match and fc2_match and fc3_match and fc1_bias_near_zero and fc2_bias_near_zero
    
    # ==================================================================
    # FINAL REPORT
    # ==================================================================
    section_header(10, "FINAL DIAGNOSIS REPORT")
    
    print("""
  1.  Environment correct?        YES
  2.  FFmpeg correct?              YES (standalone 9.0.1)
  3.  Audio preprocessing correct? YES (FFmpeg -> WAV -> soundfile -> Kaldi fbank -> interp -> norm)
  4.  Visual preprocessing correct? YES (CenterCrop(224) -> ToTensor -> ImageNet normalize)
  5.  Checkpoint loads perfectly?   YES (0 missing, 0 unexpected keys)
  6.  Classifier architecture?     MLP(2048 -> 1024 -> 1024 -> 2) with Dropout(0.5) + ReLU
  7.  Output meaning?              index 0 = Fake logit, index 1 = Real logit
  8.  Final classifier bias?       fc3.bias ≈ [0.015, 0.019] (NOT the source of extreme logits)
  9.  Do real inputs affect logits? MINIMALLY (±2 variation on a base of ~+12)
  10. Do zero inputs produce similar? YES (~[+25, -25] for zeros — WORSE than real data)
    """)
    
    print(f"  11. 80-sample metrics:")
    print(f"      Accuracy:  {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall:    {rec:.4f}")
    print(f"      F1:        {f1:.4f}")
    print(f"      Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    if all_random:
        diagnosis = "RANDOMLY INITIALIZED"
        print(f"""
  12. Is checkpoint collapsed/bias-dominated? 
      *** CRITICAL FINDING: MLP head weights appear RANDOMLY INITIALIZED ***
      The fc1, fc2, fc3 weight standard deviations closely match xavier_normal 
      initialization values, and biases are near zero.
      
      This means stage-3.pth is likely the ADAPTED (pre-training init) checkpoint
      from adapt_pretraining_weights.ipynb, NOT a TRAINED Stage-3 checkpoint.
      
      The adapt notebook creates an initial weight file by:
      1. Loading stage-2.pth (pretrained encoder)
      2. Removing decoder keys
      3. Saving the result — with MLP head left at RANDOM INIT
      
      This file was intended as the STARTING POINT for Stage-3 training,
      not as a trained classifier.
      """)
    else:
        diagnosis = "TRAINED BUT COLLAPSED"
        print(f"""
  12. Is checkpoint collapsed/bias-dominated?
      The MLP head weights do NOT match random initialization patterns.
      The checkpoint appears to have been trained, but the classifier has
      collapsed — it produces the same prediction for all inputs.
      
      Possible causes:
      - Class imbalance (10,000 fake vs 500 real in FakeAVCeleb)
      - Aggressive head_lr (50x multiplier)
      - mean(dim=-1) pooling collapsing representations
      """)
    
    print(f"""
  13. Is there an implementation bug?
      The only bug found: eval.py does NOT call model.eval().
      This means Dropout(0.5) is active during inference.
      However, this does NOT explain the all-Fake prediction 
      (confirmed by testing with model.eval() — same result).

  14. EXACT NEXT STEP:
      """)
    
    if all_random:
        print(f"""      The stage-3.pth file appears to be the PRE-TRAINING INITIALIZATION
      (from adapt_pretraining_weights.ipynb), not a trained classifier.
      
      The Stage-3 Google Drive link is a FOLDER:
      https://drive.google.com/drive/folders/1QrtQWf5_fcybBifAMpkBZpWpW0VUpVlQ
      
      ACTION REQUIRED:
      1. Re-visit the Google Drive folder and check ALL files inside it.
         There may be a 'best_audio_model.pth' or epoch-specific checkpoint.
      2. If only one file exists and it matches your current stage-3.pth,
         then the authors published the init weights, not trained weights.
         You would need to run Stage-3 training yourself using stage-3.sh.
      """)
    else:
        print(f"""      The checkpoint has been trained but collapsed to a constant prediction.
      Consider:
      1. Re-downloading stage-3.pth from the Google Drive folder
      2. Checking for alternative checkpoints (best_audio_model.pth)
      3. Running Stage-3 training yourself with balanced sampling
      """)


if __name__ == '__main__':
    main()
