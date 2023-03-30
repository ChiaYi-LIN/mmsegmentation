import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class AttentionPoolHead(BaseDecodeHead):
    def __init__(
        self,
        spacial_dim: int,
        num_heads: int,
        pretrained: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if pretrained is not None:
            self.pretrained = pretrained
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        embed_dim = self.in_channels
        output_dim = self.channels

        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def _forward_feature(self, inputs):

        x = self._transform_inputs(inputs)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(self.positional_embedding[1:, ].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map

    def forward(self, inputs):
        """Forward function."""
        _, output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


if __name__ == "__main__":
    resolution = 16
    in_dim = 1024
    out_dim = 256
    x = torch.rand([1, in_dim, resolution, resolution])
    layer = AttentionPoolHead(resolution, in_dim, 32, out_dim)
    global_feat, feature_map = layer(x)
    print(f"{x.shape} -> {feature_map.shape}")
