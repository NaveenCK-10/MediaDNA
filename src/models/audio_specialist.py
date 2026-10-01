import torch
import torch.nn as nn
from src.models.video_cav_mae import AudioEncoder, MLP

class AudioSpecialist(nn.Module):
    def __init__(self, 
        n_classes=2,
        audio_length=1024,
        mel_bins=128,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_num_heads=12,
        encoder_depth=12,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
    ):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.audio_encoder = AudioEncoder(
            audio_length=audio_length,
            mel_bins=mel_bins,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            encoder_depth=encoder_depth,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        
        # Independent classifier (no video features)
        # Audio encoder outputs (B, 512, 768) assuming audio_length=1024, mel=128, patch=16
        # 1024 * 128 / 256 = 512 patches
        self.mlp_audio = nn.Linear(768, 1024)
        self.mlp_head = MLP(input_size=1024, hidden_size=1024, num_classes=n_classes)

    def forward(self, audio):
        # audio: (B, 1024, 128)
        audio_emb = self.audio_encoder(audio) # (B, 512, 768)
        assert audio_emb.ndim == 3
        # audio_emb = (B, N_TOKENS, D_FEATURES) = (B, 512, 768)
        
        # Pool across sequence dimension (tokens)
        audio_emb = audio_emb.mean(dim=1) # (B, 768)
        
        assert audio_emb.ndim == 2
        # audio_emb = (B, D_FEATURES) = (B, 768)
        
        audio_features = self.mlp_audio(audio_emb) # (B, 1024)
        output = self.mlp_head(audio_features) # (B, 2)
        
        return output
