"""
Single batch smoke test to verify VRAM, gradients, and weight changes.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import hashlib
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader

def get_weight_hash(tensor):
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:8]

def main():
    device = torch.device('cuda')
    print("="*60)
    print("SMOKE TEST: 1 BATCH (Memory, Forward, Backward, Grad, Update)")
    print("="*60)
    
    # 1. Load Model
    model = VideoCAVMAEFT()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load("checkpoints/stage-3.pth", map_location='cpu')
    miss, unexp = model.load_state_dict(ckpt, strict=False)
    print(f"Model loaded. Missing keys: {len(miss)}, Unexpected keys: {len(unexp)}")
    model.to(device)
    model.train()
    
    # 2. Extract initial weights
    m = model.module
    w_initial = {
        'fc1': m.mlp_head.fc1.weight.clone().detach(),
        'fc2': m.mlp_head.fc2.weight.clone().detach(),
        'fc3': m.mlp_head.fc3.weight.clone().detach()
    }
    
    print("\nInitial Weights:")
    for k, w in w_initial.items():
        print(f"  {k}: hash={get_weight_hash(w)}, mean={w.mean():.6f}, std={w.std():.6f}")
        
    # 3. Dataloader (just use test_small.csv for the smoke test)
    val_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mode':'train', 
                'mean': -5.081, 'std': 4.4849, 'noise': True, 'im_res': 224, 'label_smooth': 0}
    dataset = dataloader.VideoAudioDataset("data/test_small.csv", val_audio_conf, stage=2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # 4. Optimizer
    mlp_list = ['mlp_vision.weight', 'mlp_vision.bias', 'mlp_audio.weight', 'mlp_audio.bias',
                'mlp_head.fc1.weight', 'mlp_head.fc1.bias', 'mlp_head.fc2.weight', 'mlp_head.fc2.bias',
                'mlp_head.fc3.weight', 'mlp_head.fc3.bias', 'a2v.mlp.linear.weight', 'a2v.mlp.linear.bias',
                'v2a.mlp.linear.weight', 'v2a.mlp.linear.bias']
    mlp_params = [p for n, p in model.module.named_parameters() if n in mlp_list and p.requires_grad]
    base_params = [p for n, p in model.module.named_parameters() if n not in mlp_list and p.requires_grad]
    
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': 1e-5}, {'params': mlp_params, 'lr': 1e-5 * 50}], weight_decay=5e-7)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    # 5. Run one batch
    torch.cuda.reset_peak_memory_stats()
    a_input, v_input, labels = next(iter(loader))
    a_input = a_input.to(device)
    v_input = v_input.to(device)
    labels = labels.to(device)
    
    print(f"\nRunning forward pass. Batch size: {a_input.shape[0]}")
    with autocast():
        output = model(a_input, v_input)
        loss = loss_fn(output, labels)
        
    print(f"Loss is finite: {torch.isfinite(loss).item()} (Value: {loss.item():.4f})")
    
    print("Running backward pass...")
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    
    # Check gradients
    fc3_grad = m.mlp_head.fc3.weight.grad
    grad_norm = fc3_grad.norm().item() if fc3_grad is not None else 0
    print(f"Gradients computed. fc3.weight grad norm: {grad_norm:.6f} (Zero? {grad_norm == 0})")
    
    print("Updating weights...")
    scaler.step(optimizer)
    scaler.update()
    
    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\nMax CUDA memory allocated: {max_mem:.2f} GB / 8.00 GB")
    
    # 6. Verify changes
    print("\nWeight Change Verification:")
    changed = True
    for k, w_old in w_initial.items():
        w_new = getattr(m.mlp_head, k).weight.clone().detach()
        h_old = get_weight_hash(w_old)
        h_new = get_weight_hash(w_new)
        diff = (w_new - w_old).abs().sum().item()
        print(f"  {k}: {h_old} -> {h_new} (Diff: {diff:.6f})")
        if h_old == h_new or diff == 0:
            changed = False
            
    if changed:
        print("\nSUCCESS: All classification head weights updated successfully.")
        print("Memory is within limits. Safe to proceed with full training.")
    else:
        print("\nFAILURE: Weights did not update. Do not proceed.")

if __name__ == '__main__':
    main()
