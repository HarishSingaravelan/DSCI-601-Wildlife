# modeling/detr_transformer.py

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    """
    Standard DETR sine-cosine positional encoding.
    mask: [B,H,W] where True = padding
    """

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, mask: torch.Tensor):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [B,2F,H,W]
        return pos


class DETRTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, src_key_padding_mask, query_embed, pos_embed):
        """
        src: [B,C,H,W] projected features
        src_key_padding_mask: [B,HW] True=pad
        query_embed: [Q,C]
        pos_embed: [B,C,H,W]
        returns hs: [B,Q,C] final decoder outputs
        """
        B, C, H, W = src.shape

        src = src.flatten(2).permute(2, 0, 1)      # [HW,B,C]
        pos = pos_embed.flatten(2).permute(2, 0, 1) # [HW,B,C]

        # queries
        Q = query_embed.shape[0]
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)  # [Q,B,C]
        tgt = torch.zeros_like(query_embed)  # [Q,B,C]

        memory = self.encoder(src + pos, src_key_padding_mask=src_key_padding_mask)
        hs = self.decoder(tgt, memory,
                          tgt_key_padding_mask=None,
                          memory_key_padding_mask=src_key_padding_mask,
                          tgt_mask=None,
                          memory_mask=None,
                          )
        hs = hs.permute(1, 0, 2)  # [B,Q,C]
        return hs