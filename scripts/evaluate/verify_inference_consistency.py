import torch
import torch.nn as nn
from backend.inference import OpenAVFFService
import numpy as np

# Initialize Service
service = OpenAVFFService()
model = service.model
model.eval()

# CRITICAL ISSUE 3: One Canonical Inference Function
def run_canonical_inference(audio_tensor, video_tensor):
    """
    Executes the forward pass using the rigorously verified argument order: (audio, video).
    Applies the correct Sigmoid transformation as used by BCEWithLogitsLoss.
    Maps logit 0 -> Fake, logit 1 -> Real.
    """
    # Ensure correct shape
    assert audio_tensor.dim() == 3, f"Expected 3D audio tensor (B, 1024, 128), got {audio_tensor.shape}"
    assert video_tensor.dim() == 5, f"Expected 5D video tensor (B, 3, 16, 224, 224), got {video_tensor.shape}"
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Verified Order: audio first, video second (based on VideoCAVMAEFT.forward)
            out = model(audio_tensor, video_tensor)
            
            if isinstance(out, dict):
                logits = out['pred']
            else:
                logits = out
                
    logits = logits.cpu().float()
    
    # CRITICAL ISSUE 1: Exact production calculation
    # Production uses `probabilities = torch.sigmoid(output).cpu().float().numpy()[0]`
    probs = torch.sigmoid(logits)
    
    # Fake is index 0, Real is index 1
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    
    return {
        'logits': logits.numpy()[0].tolist(),
        'fake_prob': fake_prob,
        'real_prob': real_prob,
        'margin': fake_prob - real_prob,
        'class_mapping': {0: 'FAKE', 1: 'REAL'}
    }

print("=" * 60)
print("CRITICAL ISSUE 4 & 5: ZERO-INPUT ABLATION & TENSOR TRACING")
print("=" * 60)

# Helpers to get hooks
activation_stats = {}
def get_hook(name):
    def hook(module, input, output):
        # We only care about output
        if isinstance(output, torch.Tensor):
            t = output.float()
            activation_stats[name] = {
                'shape': list(t.shape),
                'min': t.min().item(),
                'max': t.max().item(),
                'mean': t.mean().item(),
                'std': t.std().item(),
                'norm': torch.norm(t).item()
            }
    return hook

# Register hooks for tracing
target_model = model.module if hasattr(model, 'module') else model
hooks = []
hooks.append(target_model.audio_encoder.register_forward_hook(get_hook('audio_encoder')))
hooks.append(target_model.visual_encoder.register_forward_hook(get_hook('visual_encoder')))
if hasattr(target_model, 'a2v'): hooks.append(target_model.a2v.register_forward_hook(get_hook('a2v')))
if hasattr(target_model, 'v2a'): hooks.append(target_model.v2a.register_forward_hook(get_hook('v2a')))
if hasattr(target_model, 'mlp_vision'): hooks.append(target_model.mlp_vision.register_forward_hook(get_hook('mlp_vision')))
if hasattr(target_model, 'mlp_audio'): hooks.append(target_model.mlp_audio.register_forward_hook(get_hook('mlp_audio')))
if hasattr(target_model, 'mlp_head'): hooks.append(target_model.mlp_head.register_forward_hook(get_hook('mlp_head')))

# Base tensors
norm_audio = torch.randn((1, 1024, 128)).to(service.device) * 4.4849 - 5.081 # Approx normal
norm_video = torch.randn((1, 3, 16, 224, 224)).to(service.device)
zero_audio = torch.zeros((1, 1024, 128)).to(service.device)
zero_video = torch.zeros((1, 3, 16, 224, 224)).to(service.device)

cases = {
    'A (Normal Video + Normal Audio)': (norm_audio, norm_video),
    'B (Zero Video + Normal Audio)': (norm_audio, zero_video),
    'C (Normal Video + Zero Audio)': (zero_audio, norm_video),
    'D (Zero Video + Zero Audio)': (zero_audio, zero_video)
}

for name, (a_input, v_input) in cases.items():
    print(f"\n--- CASE {name} ---")
    activation_stats.clear()
    
    # Run twice for determinism verification
    res1 = run_canonical_inference(a_input, v_input)
    res2 = run_canonical_inference(a_input, v_input)
    
    assert np.allclose(res1['logits'], res2['logits'], atol=1e-5), "Determinism failed!"
    
    print(f"Video Stats: Min={v_input.min().item():.4f}, Max={v_input.max().item():.4f}")
    print(f"Audio Stats: Min={a_input.min().item():.4f}, Max={a_input.max().item():.4f}")
    print(f"Raw Logits: {res1['logits']}")
    print(f"Fake Prob: {res1['fake_prob']:.4f}")
    print(f"Real Prob: {res1['real_prob']:.4f}")
    print(f"Fake-Real Margin: {res1['margin']:.4f}")
    
    if name == 'D (Zero Video + Zero Audio)':
        print("\n--- TRACING STATS (ZERO/ZERO) ---")
        for layer, stats in activation_stats.items():
            print(f"Layer: {layer:15} | Shape: {str(stats['shape']):18} | Norm: {stats['norm']:8.4f} | Min: {stats['min']:8.4f} | Max: {stats['max']:8.4f} | Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f}")


print("\n" + "=" * 60)
print("CRITICAL ISSUE 6: CLASSIFIER BIAS MATH")
print("=" * 60)

# Extract classifier weights
fc3 = target_model.mlp_head.fc3
w = fc3.weight.data.cpu()
b = fc3.bias.data.cpu()

print(f"fc3 Weight Shape: {w.shape}")
print(f"fc3 Bias Shape: {b.shape}")
print(f"fc3 Bias Values: {b.tolist()}")

print("\n" + "=" * 60)
print("CRITICAL ISSUE 9: REAL VIDEO (naveen.mp4) RE-RUN")
print("=" * 60)

# Run canonical inference on actual video
video_path = 'src/naveen.mp4'
# Need to use the proper backend extraction methods
fbank = service._extract_audio_fbank(video_path).to(service.device).unsqueeze(0)
vframes, _ = service._extract_video_frames(video_path)
vframes = vframes.unsqueeze(0).to(service.device)

res_real = run_canonical_inference(fbank, vframes)

print("--- naveen.mp4 Results ---")
print(f"Raw Logits: {res_real['logits']}")
print(f"Class Mapping: {res_real['class_mapping']}")
print(f"Fake Prob: {res_real['fake_prob']:.4f}")

# Re-run visual anomaly
from backend.modules import analyze_visual_signals
vis = analyze_visual_signals(video_path)
v_score = vis['visual_anomaly_score']
print(f"Visual Anomaly Score: {v_score:.4f}")

fusion = (res_real['fake_prob'] * 0.8) + (v_score * 0.2)
print(f"Fusion Score: {fusion:.4f}")
print(f"Prediction (thresh=0.5): {'FAKE' if fusion >= 0.5 else 'REAL'}")

for h in hooks: h.remove()
