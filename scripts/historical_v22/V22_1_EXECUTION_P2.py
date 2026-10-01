import torch
import torch.nn as nn
import random

# Base model assumed to be VideoCAVMAEFT-like

# MODEL D: MODALITY DROPOUT
class ModelD_ModalityDropout(nn.Module):
    def __init__(self, base_model, dropout_rate=0.25):
        super().__init__()
        self.base = base_model
        self.dropout_rate = dropout_rate
        
    def forward(self, a, v):
        if self.training:
            # Randomly drop audio
            if random.random() < self.dropout_rate:
                a = torch.zeros_like(a)
            # Randomly drop visual
            if random.random() < self.dropout_rate:
                v = torch.zeros_like(v)
        return self.base(a, v)

# MODEL E: MODALITY-SPECIFIC HEADS (MULTITASK)
class ModelE_Multitask(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model # We assume base model exposes intermediate features
        self.audio_head = nn.Linear(1024, 1)
        self.video_head = nn.Linear(1024, 1)
        self.fused_head = nn.Linear(2048, 1)
        
    def forward(self, a, v):
        # Dummy feature extraction logic for demonstration
        a_feat = torch.mean(a, dim=(1,2)) # Mock extraction
        v_feat = torch.mean(v, dim=(1,2,3,4)) # Mock extraction
        
        # Ensure sizes match mock heads
        a_feat = torch.zeros(a.shape[0], 1024).to(a.device)
        v_feat = torch.zeros(v.shape[0], 1024).to(v.device)
        
        a_out = self.audio_head(a_feat)
        v_out = self.video_head(v_feat)
        fused_out = self.fused_head(torch.cat([a_feat, v_feat], dim=-1))
        
        return fused_out, a_out, v_out

# MODEL F1: LATE FUSION
class ModelF1_LateFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Independent encoders
        self.audio_encoder = nn.Linear(128, 512)
        self.video_encoder = nn.Linear(3, 512)
        self.classifier = nn.Linear(1024, 1)
        
    def forward(self, a, v):
        a_feat = torch.mean(self.audio_encoder(a), dim=(1,2))
        # v shape is B, 3, 16, 224, 224 -> permute to B, 16, 224, 224, 3
        v_perm = v.permute(0, 2, 3, 4, 1)
        v_feat = torch.mean(self.video_encoder(v_perm), dim=(1,2,3))
        return self.classifier(torch.cat([a_feat, v_feat], dim=-1))

# MODEL F2: GATED FUSION
class ModelF2_GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Linear(128, 512)
        self.video_encoder = nn.Linear(3, 512)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(1024, 2),
            nn.Softmax(dim=-1)
        )
        self.classifier = nn.Linear(512, 1)
        
    def forward(self, a, v):
        a_feat = torch.mean(self.audio_encoder(a), dim=(1,2))
        v_perm = v.permute(0, 2, 3, 4, 1)
        v_feat = torch.mean(self.video_encoder(v_perm), dim=(1,2,3))
        
        gate_weights = self.gate(torch.cat([a_feat, v_feat], dim=-1))
        
        gated_feat = (a_feat * gate_weights[:, 0:1]) + (v_feat * gate_weights[:, 1:2])
        return self.classifier(gated_feat)

# MODEL G: A/V SYNCHRONIZATION HEAD
class ModelG_AVSync(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.sync_head = nn.Linear(1024, 1) # Predicts 1 if sync, 0 if out-of-sync
        
    def forward(self, a, v, shifted_a=None):
        out = self.base(a, v)
        
        # If training, we also predict sync using the shifted audio
        if shifted_a is not None:
            sync_feat = torch.zeros(a.shape[0], 1024).to(a.device) # Mock extraction
            sync_out = self.sync_head(sync_feat)
            return out, sync_out
            
        return out
