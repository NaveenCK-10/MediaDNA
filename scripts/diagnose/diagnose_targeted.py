"""
Targeted diagnostic to isolate the cause of constant [+13, -13] logits.
Tests: autocast vs fp32, visual preprocessing, intermediate activations.
"""
import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader
import numpy as np
from torch.cuda.amp import autocast
import os

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load model
    audio_model = VideoCAVMAEFT()
    audio_model = torch.nn.DataParallel(audio_model)
    ckpt = torch.load("checkpoints/stage-3.pth", map_location='cpu')
    audio_model.load_state_dict(ckpt, strict=False)
    audio_model.eval()
    audio_model.to(device)
    
    # Dataset
    val_audio_conf = {
        'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0,
        'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224
    }
    dataset = dataloader.VideoAudioEvalDataset(csv_file="data/test_small.csv", audio_conf=val_audio_conf)
    
    print("="*70)
    print("TEST 1: autocast (fp16) vs full precision (fp32)")
    print("="*70)
    
    a_input, v_input, labels, video_name = dataset[0]
    a_input = a_input.unsqueeze(0).to(device)
    v_input = v_input.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # With autocast
        with autocast():
            out_fp16 = audio_model(a_input, v_input)
        print(f"  autocast output: {out_fp16.cpu().float().numpy()[0]}")
        
        # Without autocast (full fp32)
        out_fp32 = audio_model(a_input, v_input)
        print(f"  fp32 output:     {out_fp32.cpu().numpy()[0]}")
        
        # With random noise audio
        rand_audio = torch.randn_like(a_input)
        out_rand_a = audio_model(rand_audio, v_input)
        print(f"  random audio:    {out_rand_a.cpu().numpy()[0]}")
        
        # With random noise video
        rand_video = torch.randn_like(v_input)
        out_rand_v = audio_model(a_input, rand_video)
        print(f"  random video:    {out_rand_v.cpu().numpy()[0]}")
        
        # With both random
        out_rand_both = audio_model(rand_audio, rand_video)
        print(f"  both random:     {out_rand_both.cpu().numpy()[0]}")
        
        # With zeros
        zero_audio = torch.zeros_like(a_input)
        zero_video = torch.zeros_like(v_input)
        out_zeros = audio_model(zero_audio, zero_video)
        print(f"  both zeros:      {out_zeros.cpu().numpy()[0]}")
    
    print("\n" + "="*70)
    print("TEST 2: Intermediate activations for real sample")
    print("="*70)
    
    model = audio_model.module  # unwrap DataParallel
    model.eval()
    
    with torch.no_grad():
        audio_emb = model.audio_encoder(a_input)
        video_emb = model.visual_encoder(v_input)
        
        print(f"  audio_emb shape: {audio_emb.shape}")
        print(f"  audio_emb stats: min={audio_emb.min():.4f}, max={audio_emb.max():.4f}, mean={audio_emb.mean():.4f}, std={audio_emb.std():.4f}")
        print(f"  video_emb shape: {video_emb.shape}")
        print(f"  video_emb stats: min={video_emb.min():.4f}, max={video_emb.max():.4f}, mean={video_emb.mean():.4f}, std={video_emb.std():.4f}")
        
        b, t, c = audio_emb.shape
        audio_emb_r = audio_emb.reshape(b, model.n_frames // 2, -1, c)
        b, t, c = video_emb.shape
        video_emb_r = video_emb.reshape(b, model.n_frames // 2, -1, c)
        
        print(f"\n  audio_emb reshaped: {audio_emb_r.shape}")
        print(f"  video_emb reshaped: {video_emb_r.shape}")
        
        video_fusion = model.a2v(audio_emb_r)
        audio_fusion = model.v2a(video_emb_r)
        
        print(f"\n  a2v output stats: min={video_fusion.min():.4f}, max={video_fusion.max():.4f}, mean={video_fusion.mean():.4f}, std={video_fusion.std():.4f}")
        print(f"  v2a output stats: min={audio_fusion.min():.4f}, max={audio_fusion.max():.4f}, mean={audio_fusion.mean():.4f}, std={audio_fusion.std():.4f}")
        
        # Concat
        video_cat = torch.concat((video_fusion, video_emb_r), dim=-1)
        audio_cat = torch.concat((audio_fusion, audio_emb_r), dim=-1)
        
        print(f"\n  video after concat: {video_cat.shape}")
        print(f"  audio after concat: {audio_cat.shape}")
        
        video_pooled = video_cat.mean(dim=-1)
        audio_pooled = audio_cat.mean(dim=-1)
        
        print(f"  video after mean(dim=-1): {video_pooled.shape}, stats: min={video_pooled.min():.6f}, max={video_pooled.max():.6f}, mean={video_pooled.mean():.6f}")
        print(f"  audio after mean(dim=-1): {audio_pooled.shape}, stats: min={audio_pooled.min():.6f}, max={audio_pooled.max():.6f}, mean={audio_pooled.mean():.6f}")
        
        from einops import rearrange
        video_flat = rearrange(video_pooled, 'b t c -> b (t c)')
        audio_flat = rearrange(audio_pooled, 'b t c -> b (t c)')
        
        print(f"\n  video_flat: {video_flat.shape}, stats: min={video_flat.min():.6f}, max={video_flat.max():.6f}")
        print(f"  audio_flat: {audio_flat.shape}, stats: min={audio_flat.min():.6f}, max={audio_flat.max():.6f}")
        
        video_proj = model.mlp_vision(video_flat)
        audio_proj = model.mlp_audio(audio_flat)
        
        print(f"\n  video after mlp_vision: stats: min={video_proj.min():.6f}, max={video_proj.max():.6f}, mean={video_proj.mean():.6f}")
        print(f"  audio after mlp_audio: stats: min={audio_proj.min():.6f}, max={audio_proj.max():.6f}, mean={audio_proj.mean():.6f}")
        
        combined = torch.concat((video_proj, audio_proj), dim=-1)
        print(f"\n  combined input to mlp_head: {combined.shape}")
        
        # Step through MLP head
        x = model.mlp_head.fc1(combined)
        print(f"  after fc1: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
        x = model.mlp_head.act1(x)
        print(f"  after relu1: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, nonzero={(x > 0).sum()}/{x.numel()}")
        x = model.mlp_head.fc2(x)
        print(f"  after fc2: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
        x = model.mlp_head.act2(x)
        print(f"  after relu2: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, nonzero={(x > 0).sum()}/{x.numel()}")
        x = model.mlp_head.fc3(x)
        print(f"  after fc3 (final logits): {x.cpu().numpy()[0]}")
    
    print("\n" + "="*70)
    print("TEST 3: Check fc3 weight statistics")
    print("="*70)
    fc3_w = model.mlp_head.fc3.weight.data
    fc3_b = model.mlp_head.fc3.bias.data
    print(f"  fc3 weight shape: {fc3_w.shape}")
    print(f"  fc3 weight[0] (fake logit): min={fc3_w[0].min():.6f}, max={fc3_w[0].max():.6f}, mean={fc3_w[0].mean():.6f}, std={fc3_w[0].std():.6f}")
    print(f"  fc3 weight[1] (real logit): min={fc3_w[1].min():.6f}, max={fc3_w[1].max():.6f}, mean={fc3_w[1].mean():.6f}, std={fc3_w[1].std():.6f}")
    print(f"  fc3 bias: {fc3_b.cpu().numpy()}")
    
    # Check if weights are nearly identical (which would explain similar outputs)
    weight_diff = (fc3_w[0] - fc3_w[1]).abs()
    print(f"  |w[0] - w[1]|: min={weight_diff.min():.6f}, max={weight_diff.max():.6f}, mean={weight_diff.mean():.6f}")
    
    # Check if fc3_w[0] ≈ -fc3_w[1] (anti-correlated)
    correlation = torch.corrcoef(fc3_w)[0, 1]
    print(f"  correlation between w[0] and w[1]: {correlation:.6f}")
    
    print("\n" + "="*70)
    print("TEST 4: Compare real vs fake video — intermediate differences")
    print("="*70)
    
    # Get a real and fake sample
    fake_a, fake_v, fake_l, fake_name = dataset[0]  # FakeVideo-FakeAudio, label 1
    real_a, real_v, real_l, real_name = dataset[2]   # RealVideo-RealAudio, label 0
    
    fake_a = fake_a.unsqueeze(0).to(device)
    fake_v = fake_v.unsqueeze(0).to(device)
    real_a = real_a.unsqueeze(0).to(device)
    real_v = real_v.unsqueeze(0).to(device)
    
    with torch.no_grad():
        fake_out = model(fake_a, fake_v)
        real_out = model(real_a, real_v)
        
        print(f"  Fake video logits: {fake_out.cpu().numpy()[0]}")
        print(f"  Real video logits: {real_out.cpu().numpy()[0]}")
        print(f"  Difference: {(fake_out - real_out).cpu().numpy()[0]}")
        
        # Compare encoder outputs
        fake_audio_emb = model.audio_encoder(fake_a)
        real_audio_emb = model.audio_encoder(real_a)
        fake_video_emb = model.visual_encoder(fake_v)
        real_video_emb = model.visual_encoder(real_v)
        
        print(f"\n  Fake audio_emb mean: {fake_audio_emb.mean():.6f}")
        print(f"  Real audio_emb mean: {real_audio_emb.mean():.6f}")
        print(f"  Audio emb mean diff: {(fake_audio_emb.mean() - real_audio_emb.mean()):.6f}")
        
        print(f"\n  Fake video_emb mean: {fake_video_emb.mean():.6f}")
        print(f"  Real video_emb mean: {real_video_emb.mean():.6f}")
        print(f"  Video emb mean diff: {(fake_video_emb.mean() - real_video_emb.mean()):.6f}")
        
        # Cosine similarity of final combined representations
        def get_combined(a_in, v_in):
            ae = model.audio_encoder(a_in)
            ve = model.visual_encoder(v_in)
            b, t, c = ae.shape
            ae = ae.reshape(b, model.n_frames // 2, -1, c)
            b, t, c = ve.shape
            ve = ve.reshape(b, model.n_frames // 2, -1, c)
            vf = model.a2v(ae)
            af = model.v2a(ve)
            vc = torch.concat((vf, ve), dim=-1)
            ac = torch.concat((af, ae), dim=-1)
            vc = vc.mean(dim=-1)
            ac = ac.mean(dim=-1)
            from einops import rearrange
            vc = rearrange(vc, 'b t c -> b (t c)')
            ac = rearrange(ac, 'b t c -> b (t c)')
            vp = model.mlp_vision(vc)
            ap = model.mlp_audio(ac)
            return torch.concat((vp, ap), dim=-1)
        
        fake_repr = get_combined(fake_a, fake_v)
        real_repr = get_combined(real_a, real_v)
        
        cos_sim = nn.functional.cosine_similarity(fake_repr, real_repr)
        print(f"\n  Cosine similarity of fake vs real combined repr: {cos_sim.item():.6f}")
        print(f"  L2 distance: {(fake_repr - real_repr).norm():.6f}")
        print(f"  fake repr norm: {fake_repr.norm():.6f}")
        print(f"  real repr norm: {real_repr.norm():.6f}")


if __name__ == '__main__':
    main()
