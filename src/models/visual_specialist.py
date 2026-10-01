import torch
import torch.nn as nn
from einops import rearrange
from src.models.video_cav_mae import VisualEncoder, MLP

class VisualSpecialist(nn.Module):
    def __init__(self, 
        n_classes=2,
        img_size=224,
        patch_size=16, 
        n_frames=16, 
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        norm_layer="LayerNorm",
        init_values=0.,
        tubelet_size=2
    ):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_frames = n_frames
        
        # Exact V22.2 visual encoder
        self.visual_encoder = VisualEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            n_frames=n_frames, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )
        
        # Independent classifier (no audio features)
        # Visual encoder outputs (B, 1568, 768)
        # We will pool across the sequence dimension (1568)
        self.mlp_vision = nn.Linear(768, 1024)
        self.mlp_head = MLP(input_size=1024, hidden_size=1024, num_classes=n_classes)

    def forward(self, video):
        # video: (B, 3, 16, 224, 224)
        video_emb = self.visual_encoder(video) # (B, 1568, 768)
        
        assert video_emb.ndim == 3
        # video_emb = (B, N_TOKENS, D_FEATURES) = (B, 1568, 768)
        
        # Pool across sequence dimension (tokens)
        video_emb = video_emb.mean(dim=1) # (B, 768)
        
        assert video_emb.ndim == 2
        # video_emb = (B, D_FEATURES) = (B, 768)
        
        video_features = self.mlp_vision(video_emb) # (B, 1024)
        output = self.mlp_head(video_features) # (B, 2)
        
        return output
