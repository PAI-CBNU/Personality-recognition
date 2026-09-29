# -*- coding: utf-8 -*-

import math
from functools import lru_cache, reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from resnet import r2plus1d_18


# ============================================================
# Basic Components
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ============================================================
# Swin Window Utilities
# ============================================================

def window_partition(x, window_size):
    """
    x: [B, D, H, W, C]
    return: [B*num_windows, Wd*Wh*Ww, C]
    """
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size

    x = x.view(
        B, D // Wd, Wd,
        H // Wh, Wh,
        W // Ww, Ww, C
    )

    return x.permute(
        0, 1, 3, 5, 2, 4, 6, 7
    ).contiguous().view(-1, reduce(mul, window_size), C)


def window_reverse(windows, window_size, B, D, H, W):
    Wd, Wh, Ww = window_size

    x = windows.view(
        B, D // Wd, H // Wh, W // Ww,
        Wd, Wh, Ww, -1
    )

    return x.permute(
        0, 1, 4, 2, 5, 3, 6, 7
    ).contiguous().view(B, D, H, W, -1)


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None

    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]

            if use_shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)

    return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)

    d_slices = (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None)
    )

    h_slices = (
        slice(-window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None)
    )

    w_slices = (
        slice(-window_size[2]),
        slice(-window_size[2], -shift_size[2]),
        slice(-shift_size[2], None)
    )

    count = 0

    for d in d_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, d, h, w, :] = count
                count += 1

    mask_windows = window_partition(
        img_mask, window_size
    ).squeeze(-1)

    attn_mask = (
        mask_windows.unsqueeze(1)
        - mask_windows.unsqueeze(2)
    )

    attn_mask = attn_mask.masked_fill(
        attn_mask != 0, -100.0
    ).masked_fill(
        attn_mask == 0, 0.0
    )

    return attn_mask


# ============================================================
# 3D Window Attention
# ============================================================

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        bias_size = (
            (2 * window_size[0] - 1) *
            (2 * window_size[1] - 1) *
            (2 * window_size[2] - 1)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(bias_size, num_heads)
        )

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])

        coords = torch.stack(
            torch.meshgrid(
                coords_d, coords_h, coords_w,
                indexing="ij"
            )
        )

        coords_flatten = torch.flatten(coords, 1)

        relative_coords = (
            coords_flatten[:, :, None]
            - coords_flatten[:, None, :]
        ).permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (
            (2 * window_size[1] - 1)
            * (2 * window_size[2] - 1)
        )

        relative_coords[:, :, 1] *= (
            2 * window_size[2] - 1
        )

        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1)
        )

        self.qkv = nn.Linear(
            dim, dim * 3, bias=qkv_bias
        )

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(
            self.relative_position_bias_table,
            std=.02
        )

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B_, N, 3,
            self.num_heads,
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(N, N, -1)

        relative_bias = relative_bias.permute(
            2, 0, 1
        ).contiguous()

        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]

            attn = attn.view(
                B_ // nW,
                nW,
                self.num_heads,
                N,
                N
            )

            attn = attn + mask.unsqueeze(0).unsqueeze(2)

            attn = attn.view(
                -1,
                self.num_heads,
                N,
                N
            )

        attn = self.attn_drop(
            self.softmax(attn)
        )

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(B_, N, C)
        )

        return self.proj_drop(
            self.proj(x)
        )


# ============================================================
# Swin Transformer Block
# ============================================================

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7),
                 shift_size=(0, 0, 0), mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        for shift, window in zip(
            shift_size, window_size
        ):
            assert 0 <= shift < window

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop
        )

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape

        window_size, shift_size = get_window_size(
            (D, H, W),
            self.window_size,
            self.shift_size
        )

        x = self.norm1(x)

        pad_d = (
            window_size[0]
            - D % window_size[0]
        ) % window_size[0]

        pad_h = (
            window_size[1]
            - H % window_size[1]
        ) % window_size[1]

        pad_w = (
            window_size[2]
            - W % window_size[2]
        ) % window_size[2]

        x = F.pad(
            x,
            (0, 0, 0, pad_w, 0, pad_h, 0, pad_d)
        )

        _, Dp, Hp, Wp, _ = x.shape

        if any(s > 0 for s in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=tuple(
                    -s for s in shift_size
                ),
                dims=(1, 2, 3)
            )

            attn_mask = mask_matrix

        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(
            shifted_x,
            window_size
        )

        attn_windows = self.attn(
            x_windows,
            mask=attn_mask
        )

        attn_windows = attn_windows.view(
            -1, *window_size, C
        )

        shifted_x = window_reverse(
            attn_windows,
            window_size,
            B, Dp, Hp, Wp
        )

        if any(s > 0 for s in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=shift_size,
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        return x[:, :D, :H, :W, :].contiguous()

    def forward_part2(self, x):
        return self.drop_path(
            self.mlp(self.norm2(x))
        )

    def forward(self, x, mask_matrix):
        shortcut = x

        if self.use_checkpoint:
            x = checkpoint.checkpoint(
                self.forward_part1,
                x,
                mask_matrix
            )
        else:
            x = self.forward_part1(
                x,
                mask_matrix
            )

        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(
                self.forward_part2,
                x
            )
        else:
            x = x + self.forward_part2(x)

        return x


# ============================================================
# Patch Embedding / Patch Merging
# ============================================================

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_chans=512,
                 embed_dim=96, norm_layer=None):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = (
            norm_layer(embed_dim)
            if norm_layer is not None
            else None
        )

    def forward(self, x):
        _, _, D, H, W = x.shape

        pad_w = (
            self.patch_size[2]
            - W % self.patch_size[2]
        ) % self.patch_size[2]

        pad_h = (
            self.patch_size[1]
            - H % self.patch_size[1]
        ) % self.patch_size[1]

        pad_d = (
            self.patch_size[0]
            - D % self.patch_size[0]
        ) % self.patch_size[0]

        if pad_w > 0:
            x = F.pad(x, (0, pad_w))

        if pad_h > 0:
            x = F.pad(
                x, (0, 0, 0, pad_h)
            )

        if pad_d > 0:
            x = F.pad(
                x, (0, 0, 0, 0, 0, pad_d)
            )

        x = self.proj(x)

        if self.norm is not None:
            D, H, W = x.shape[2:]

            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)

            x = x.transpose(1, 2).reshape(
                -1,
                self.embed_dim,
                D, H, W
            )

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm = norm_layer(4 * dim)

        self.reduction = nn.Linear(
            4 * dim,
            2 * dim,
            bias=False
        )

    def forward(self, x):
        _, _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(
                x,
                (0, 0, 0, W % 2, 0, H % 2)
            )

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]

        x = torch.cat(
            [x0, x1, x2, x3],
            dim=-1
        )

        return self.reduction(
            self.norm(x)
        )


# ============================================================
# One Video Swin Stage
# ============================================================

class BasicLayer(nn.Module):
    """
    Pure Video Swin stage:

        Swin Block 1
            ↓
        Swin Block 2
            ↓
           ...
            ↓
        Patch Merging

    No audio branch.
    No Cross-Attention.
    """

    def __init__(self, dim, depth, num_heads, window_size=(2, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=PatchMerging,
                 use_checkpoint=False):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.window_size = window_size
        self.shift_size = tuple(
            w // 2
            for w in window_size
        )

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(
                    (0, 0, 0)
                    if i % 2 == 0
                    else self.shift_size
                ),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=(
                    drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path
                ),
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)
        ])

        self.downsample = (
            downsample(
                dim=dim,
                norm_layer=norm_layer
            )
            if downsample is not None
            else None
        )

    def forward(self, video, return_features=False):
        B, C, D, H, W = video.shape

        window_size, shift_size = get_window_size(
            (D, H, W),
            self.window_size,
            self.shift_size
        )

        video = rearrange(
            video,
            "b c d h w -> b d h w c"
        )

        Dp = int(
            np.ceil(D / window_size[0])
        ) * window_size[0]

        Hp = int(
            np.ceil(H / window_size[1])
        ) * window_size[1]

        Wp = int(
            np.ceil(W / window_size[2])
        ) * window_size[2]

        attn_mask = compute_mask(
            Dp, Hp, Wp,
            window_size,
            shift_size,
            video.device
        )

        block_outputs = []

        # ----------------------------------------------------
        # Pure Video Swin Blocks
        # ----------------------------------------------------
        for block in self.blocks:
            video = block(
                video,
                attn_mask
            )

            if return_features:
                block_outputs.append(video)

        # ----------------------------------------------------
        # Patch Merging
        # ----------------------------------------------------
        if self.downsample is not None:
            video = self.downsample(video)

        stage_output = (
            video
            if return_features
            else None
        )

        video = rearrange(
            video,
            "b d h w c -> b c d h w"
        )

        if return_features:
            return (
                video,
                block_outputs,
                stage_output
            )

        return video


# ============================================================
# Full Video-only Swin Transformer
# ============================================================

class SwinTransformer3D(nn.Module):
    def __init__(
        self,

        # Video Swin
        patch_size=(4, 4, 4),
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(2, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,

        # Regularization
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,

        # Other
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        use_checkpoint=False,
        frozen_stages=-1,

        # Big Five regression
        head_hidden_dim=512,
        num_outputs=5
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.depths = tuple(depths)
        self.frozen_stages = frozen_stages

        # ====================================================
        # Video Backbone
        # ====================================================
        self.video_backbone = r2plus1d_18()

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=512,
            embed_dim=embed_dim,
            norm_layer=(
                norm_layer
                if patch_norm
                else None
            )
        )

        self.pos_drop = nn.Dropout(
            drop_rate
        )

        # ====================================================
        # Stochastic Depth
        # ====================================================
        dpr = [
            x.item()
            for x in torch.linspace(
                0,
                drop_path_rate,
                sum(depths)
            )
        ]

        # ====================================================
        # Video Swin Stages
        # ====================================================
        self.layers = nn.ModuleList()

        start = 0

        for stage_idx in range(
            self.num_layers
        ):
            end = (
                start
                + depths[stage_idx]
            )

            layer = BasicLayer(
                dim=embed_dim * 2 ** stage_idx,
                depth=depths[stage_idx],
                num_heads=num_heads[stage_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[start:end],
                norm_layer=norm_layer,

                # Keep exactly the same behavior
                # as your VST-IBCA version.
                downsample=PatchMerging,

                use_checkpoint=use_checkpoint
            )

            self.layers.append(layer)

            start = end

        # 96 -> 192 -> 384 -> 768 -> 1536
        self.num_features = (
            embed_dim
            * 2 ** self.num_layers
        )

        self.norm = norm_layer(
            self.num_features
        )

        # ====================================================
        # Big Five Regression Head
        # ====================================================
        self.head = nn.Sequential(
            nn.Linear(
                self.num_features,
                head_hidden_dim
            ),
            nn.Linear(
                head_hidden_dim,
                num_outputs
            )
        )

        self._freeze_stages()

    # ========================================================
    # Freeze Stages
    # ========================================================

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()

            for param in (
                self.patch_embed.parameters()
            ):
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()

            for i in range(
                min(
                    self.frozen_stages,
                    len(self.layers)
                )
            ):
                self.layers[i].eval()

                for param in (
                    self.layers[i].parameters()
                ):
                    param.requires_grad = False

    # ========================================================
    # Normal Forward
    # ========================================================

    def forward(self, video):
        """
        video:
            [B, 3, T, H, W]

        output:
            [B, 5]
        """

        # ----------------------------------------------------
        # R(2+1)D
        # ----------------------------------------------------
        video = self.video_backbone(
            video
        )

        # ----------------------------------------------------
        # 3D Patch Embedding
        # ----------------------------------------------------
        video = self.patch_embed(
            video
        )

        video = self.pos_drop(
            video
        )

        # ----------------------------------------------------
        # Pure Video Swin
        # ----------------------------------------------------
        for layer in self.layers:
            video = layer(
                video.contiguous()
            )

        # ----------------------------------------------------
        # Big Five Regression
        # ----------------------------------------------------
        video = rearrange(
            video,
            "b c t h w -> b t h w c"
        )

        video = self.norm(
            video
        )

        B, T, H, W, C = video.shape

        if T * H * W != 1:
            raise RuntimeError(
                f"Final feature size is "
                f"T={T}, H={H}, W={W}. "
                f"The current regression head "
                f"expects T*H*W = 1."
            )

        video = video.reshape(
            B, C
        )

        return self.head(video)

    # ========================================================
    # Probe Feature Extraction
    # ========================================================

    def extract_probe_features(self, video):
        """
        Extract intermediate representations.

        Returns:
            stage1_block1
            stage1_block2
            stage1
            ...
            stage4_block2
            stage4

        All returned representations are GAP pooled:
            [B,T,H,W,C] -> [B,C]
        """

        features = {}

        # ----------------------------------------------------
        # Video Backbone
        # ----------------------------------------------------
        video = self.video_backbone(
            video
        )

        video = self.patch_embed(
            video
        )

        video = self.pos_drop(
            video
        )

        # ----------------------------------------------------
        # Stage / Block Features
        # ----------------------------------------------------
        for stage_idx, layer in enumerate(
            self.layers
        ):
            (
                video,
                block_outputs,
                stage_output
            ) = layer(
                video.contiguous(),
                return_features=True
            )

            # Block-level probes
            for block_idx, feature in enumerate(
                block_outputs
            ):
                # [B,T,H,W,C] -> [B,C]
                pooled = feature.mean(
                    dim=(1, 2, 3)
                )

                key = (
                    f"stage{stage_idx + 1}_"
                    f"block{block_idx + 1}"
                )

                features[key] = pooled

            # Stage-level probe:
            # AFTER PatchMerging
            pooled_stage = stage_output.mean(
                dim=(1, 2, 3)
            )

            features[
                f"stage{stage_idx + 1}"
            ] = pooled_stage

        return features

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self
# ============================================================
# Video-only VST Linear Probe
# ============================================================

class VideoOnlyVSTProbe(nn.Module):
    """
    Frozen Video-only VST + independent linear probe heads.

    Backbone:
        completely frozen

    Trainable:
        probe_heads only

    Probe positions:
        - every Swin block output
        - every stage output after PatchMerging
    """

    def __init__(self, backbone, embed_dim=96, depths=(2, 2, 6, 2),
                 num_outputs=5, probe_blocks=True, probe_stages=True):
        super().__init__()

        self.backbone = backbone
        self.probe_blocks = probe_blocks
        self.probe_stages = probe_stages

        # ----------------------------------------------------
        # Freeze the entire pretrained Video-only VST
        # ----------------------------------------------------
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

        # ----------------------------------------------------
        # Independent probe heads
        # ----------------------------------------------------
        self.probe_heads = nn.ModuleDict()

        for stage_idx, depth in enumerate(depths):

            # Before PatchMerging:
            # Stage1 blocks: 96
            # Stage2 blocks: 192
            # Stage3 blocks: 384
            # Stage4 blocks: 768
            block_dim = embed_dim * 2 ** stage_idx

            if probe_blocks:
                for block_idx in range(depth):
                    name = f"stage{stage_idx + 1}_block{block_idx + 1}"

                    self.probe_heads[name] = nn.Linear(
                        block_dim,
                        num_outputs
                    )

            # After PatchMerging:
            # Stage1: 192
            # Stage2: 384
            # Stage3: 768
            # Stage4: 1536
            if probe_stages:
                stage_dim = block_dim * 2

                self.probe_heads[
                    f"stage{stage_idx + 1}"
                ] = nn.Linear(
                    stage_dim,
                    num_outputs
                )

    def forward(self, video):
        """
        video:
            [B, 3, T, H, W]

        returns:
            {
                "stage1_block1": [B,5],
                "stage1_block2": [B,5],
                "stage1":        [B,5],
                ...
            }
        """

        # Backbone never receives gradients
        with torch.no_grad():
            features = self.backbone.extract_probe_features(video)

        predictions = {
            name: head(features[name])
            for name, head in self.probe_heads.items()
        }

        return predictions

    def train(self, mode=True):
        """
        Only probe heads enter training mode.
        Frozen backbone always remains eval().
        """
        super().train(mode)
        self.backbone.eval()
        return self