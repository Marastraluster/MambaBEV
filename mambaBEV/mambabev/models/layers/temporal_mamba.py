"""TemporalMamba fusion (mambaBEV).

This module implements the TemporalMamba fusion pipeline described in the
mambaBEV paper.

Design constraints:
- No dependencies on mmdet/mmdet3d/mmengine.
- Internal deps are restricted to:
  - Mamba2Block
  - bev_to_sequences / sequences_to_bev
  - align_bev_by_yaw
- Works with fp16/fp32 without forcing casts.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .mamba2_wrapper import Mamba2Block
from .query_rearrange import bev_to_sequences, sequences_to_bev
from ..utils.ego_align import align_bev_by_yaw


class TemporalMamba(nn.Module):
	"""TemporalMamba fusion block.

	forward:
		curr_bev: (B, H*W, C)
		prev_bev: (B, H*W, C) or None
		returns:  (B, H*W, C)
	"""

	def __init__(
		self,
		embed_dim: int = 256,
		mamba_dim: int = 256,
		d_state: int = 128,
		d_conv: int = 4,
		expand: int = 2,
		conv_kernel: int = 3,
		dropout: float = 0.9,
		act_layer: str = "relu",
	) -> None:
		super().__init__()

		assert isinstance(embed_dim, int) and embed_dim > 0, "embed_dim must be a positive int"
		assert isinstance(mamba_dim, int) and mamba_dim > 0, "mamba_dim must be a positive int"
		assert isinstance(d_state, int) and d_state > 0, "d_state must be a positive int"
		assert isinstance(d_conv, int) and d_conv > 0, "d_conv must be a positive int"
		assert isinstance(expand, int) and expand > 0, "expand must be a positive int"
		assert isinstance(conv_kernel, int) and conv_kernel > 0, "conv_kernel must be a positive int"
		assert isinstance(dropout, float) and 0.0 <= dropout <= 1.0, "dropout must be in [0, 1]"
		assert embed_dim % 2 == 0, "embed_dim must be even because conv branches use C/2 channels"

		act = act_layer.lower()
		assert act in {"relu", "gelu"}, "act_layer only supports 'relu' or 'gelu'"

		self.embed_dim = embed_dim
		self.mamba_dim = mamba_dim

		# (论文步骤 4) 并行卷积分支：3x3 + 1x1（带 BN），输出各 C/2
		in_ch = 2 * embed_dim
		out_ch_half = embed_dim // 2
		pad = conv_kernel // 2
		self.branch3 = nn.Sequential(
			nn.Conv2d(in_ch, out_ch_half, kernel_size=conv_kernel, padding=pad, bias=False),
			nn.BatchNorm2d(out_ch_half),
		)
		self.branch1 = nn.Sequential(
			nn.Conv2d(in_ch, out_ch_half, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(out_ch_half),
		)

		self.activation = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()

		# (论文步骤 5) Linear + LayerNorm（作用在最后一维 C）
		self.fuse_linear = nn.Linear(embed_dim, embed_dim)
		self.fuse_ln = nn.LayerNorm(embed_dim)

		# (论文步骤 6) 四方向序列：每路都有 in_proj / mamba / out_proj
		# 注：这里为每个方向使用独立参数，以严格对应“对每个 seq 执行”。
		self.in_proj = nn.ModuleList([nn.Linear(embed_dim, mamba_dim) for _ in range(4)])
		self.mamba = nn.ModuleList(
			[
				Mamba2Block(
					d_model=mamba_dim,
					d_state=d_state,
					d_conv=d_conv,
					expand=expand,
					dropout=0.0,  # 注意：Mamba block 内部 dropout 设为 0
				)
				for _ in range(4)
			]
		)
		self.out_proj = nn.ModuleList([nn.Linear(mamba_dim, embed_dim) for _ in range(4)])

		# (论文步骤 8) residual dropout，只作用于 fused 分支
		self.resid_dropout = nn.Dropout(p=dropout)

	def forward(
		self,
		curr_bev: torch.Tensor,  # (B, H*W, C)
		prev_bev: Optional[torch.Tensor],  # (B, H*W, C) or None
		bev_h: int,
		bev_w: int,
		delta_yaw: Optional[torch.Tensor] = None,  # (B,) or None
	) -> torch.Tensor:
		# (论文步骤 1) shape 校验
		assert isinstance(curr_bev, torch.Tensor), "curr_bev must be a torch.Tensor"
		assert curr_bev.ndim == 3, f"curr_bev must be 3D (B,L,C), got {tuple(curr_bev.shape)}"
		b, l, c = curr_bev.shape
		assert c == self.embed_dim, f"curr_bev C must equal embed_dim={self.embed_dim}, got C={c}"
		assert isinstance(bev_h, int) and isinstance(bev_w, int) and bev_h > 0 and bev_w > 0, "bev_h/bev_w must be positive ints"
		assert bev_h * bev_w == l, f"Expected L=H*W={bev_h}*{bev_w}={bev_h * bev_w}, got L={l}"

		if prev_bev is None:
			# 无历史帧直接返回当前帧
			return curr_bev

		assert isinstance(prev_bev, torch.Tensor), "prev_bev must be a torch.Tensor or None"
		assert prev_bev.ndim == 3, f"prev_bev must be 3D (B,L,C), got {tuple(prev_bev.shape)}"
		assert prev_bev.shape == curr_bev.shape, (
			f"prev_bev shape must match curr_bev: got {tuple(prev_bev.shape)} vs {tuple(curr_bev.shape)}"
		)

		# (论文步骤 2) prev 对齐（可选 yaw 旋转）
		curr = curr_bev.view(b, bev_h, bev_w, c)
		prev = prev_bev.view(b, bev_h, bev_w, c)

		if delta_yaw is None:
			aligned_prev = prev
		else:
			# align_bev_by_yaw: CCW positive (radians). Internally uses inverse warp for grid_sample.
			aligned_prev = align_bev_by_yaw(prev, delta_yaw)

		# (论文步骤 3) concat 融合：cat([curr, aligned_prev], dim=-1) -> (B,H,W,2C)
		cat = torch.cat([curr, aligned_prev], dim=-1)
		# 转为 NCHW: (B,2C,H,W)
		cat_nchw = cat.permute(0, 3, 1, 2).contiguous()

		# (论文步骤 4) 并行卷积分支（3x3 + 1x1，带 BN）
		branch3 = self.branch3(cat_nchw)
		branch1 = self.branch1(cat_nchw)
		z = torch.cat([branch3, branch1], dim=1)  # (B,C,H,W)
		z = self.activation(z)
		# 回到 BHWC
		z = z.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)

		# (论文步骤 5) Linear + LayerNorm（最后一维 C）
		z = self.fuse_linear(z)
		z = self.fuse_ln(z)
		# 这里 z 即论文中的 Z

		# (论文步骤 6) 四方向重排 + Mamba
		seq_fl, seq_fu, seq_rl, seq_ru = bev_to_sequences(z, bev_h, bev_w)
		seqs = (seq_fl, seq_fu, seq_rl, seq_ru)

		processed = []
		for i, seq in enumerate(seqs):
			# seq: (B, L, C)
			s = self.in_proj[i](seq)
			s = self.mamba[i](s)
			s = self.out_proj[i](s)
			processed.append(s)

		seqs2 = (processed[0], processed[1], processed[2], processed[3])

		# (论文步骤 7) inverse + 四路平均（内部会做 inverse 并平均）
		fused = sequences_to_bev(seqs2, bev_h, bev_w)  # (B,H,W,C)

		# (论文步骤 8) residual + dropout：只对 fused 分支做 dropout
		fused_flat = fused.view(b, l, c)
		out = curr_bev + self.resid_dropout(fused_flat)
		assert out.shape == curr_bev.shape
		return out

