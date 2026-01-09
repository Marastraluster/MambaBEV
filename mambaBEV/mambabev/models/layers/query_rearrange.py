"""Query rearrangement utilities (four-direction, no zig-zag).

This file implements the “four-direction Query Re-arrange / Re-merge” module used
by TemporalMamba in mambaBEV.

Key idea:
- Given BEV feature Z of shape (B, H, W, C) (or flattened as (B, H*W, C)), we
  build four 1D sequences (B, L, C), L = H*W, each corresponding to a strict
  scan order (no snake/zig-zag).
- The inverse maps each sequence back to (B, H, W, C) via scatter, then averages
  the four reconstructions element-wise.

Dependencies: torch only.
"""

from __future__ import annotations

from typing import Tuple

import torch

Tensor = torch.Tensor


def _as_bev_4d(z: Tensor, bev_h: int, bev_w: int) -> Tensor:
	"""Normalize input to (B, H, W, C).

	Accepts:
	  - (B, H, W, C)
	  - (B, H*W, C)
	"""
	assert isinstance(z, torch.Tensor), "z must be a torch.Tensor"
	assert isinstance(bev_h, int) and bev_h > 0, "bev_h must be a positive int"
	assert isinstance(bev_w, int) and bev_w > 0, "bev_w must be a positive int"

	if z.ndim == 4:
		b, h, w, c = z.shape
		assert h == bev_h and w == bev_w, (
			f"Input z is (B,H,W,C)={tuple(z.shape)} but bev_h/bev_w=({bev_h},{bev_w})"
		)
		return z

	if z.ndim == 3:
		b, l, c = z.shape
		assert l == bev_h * bev_w, (
			f"Flattened input z has L={l}, expected H*W={bev_h}*{bev_w}={bev_h * bev_w}"
		)
		return z.reshape(b, bev_h, bev_w, c)

	assert False, f"z must be 3D (B,L,C) or 4D (B,H,W,C), got z.ndim={z.ndim}"


def _build_order_indices(bev_h: int, bev_w: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
	"""Build 4 direction order indices (each is LongTensor of shape (L,)).

	We always flatten the BEV as row-major (y-major then x):
		base_index(y, x) = y * W + x

	Then each direction is represented as an *order list* of these base indices.

	Directions (strict definitions):
	  1) forward-left  (row-major forward):
		 y: 0->H-1, x: 0->W-1
	  2) forward-upward (col-major forward):
		 x: 0->W-1, y: 0->H-1
	  3) reverse-left  (row-major reverse):
		 y: H-1->0, x: W-1->0
	  4) reverse-upward (col-major reverse):
		 x: W-1->0, y: H-1->0

	Returns:
		(order_fl, order_fu, order_rl, order_ru)
	"""
	h = bev_h
	w = bev_w

	# NOTE: all indices must be torch.Tensor on the correct device.
	y = torch.arange(h, device=device, dtype=torch.long)
	x = torch.arange(w, device=device, dtype=torch.long)
	y_rev = torch.arange(h - 1, -1, -1, device=device, dtype=torch.long)
	x_rev = torch.arange(w - 1, -1, -1, device=device, dtype=torch.long)

	# forward-left: (y, x) with y-major then x-major
	yy_fl, xx_fl = torch.meshgrid(y, x, indexing="ij")  # (H, W)
	order_fl = (yy_fl * w + xx_fl).reshape(-1)  # (L,)

	# forward-upward: (x, y) with x-major then y-major
	# Meshgrid with indexing="ij" gives (W, H) when inputs are (x, y).
	xx_fu, yy_fu = torch.meshgrid(x, y, indexing="ij")  # (W, H)
	order_fu = (yy_fu * w + xx_fu).reshape(-1)  # (L,)

	# reverse-left: reverse both y and x in row-major
	yy_rl, xx_rl = torch.meshgrid(y_rev, x_rev, indexing="ij")  # (H, W)
	order_rl = (yy_rl * w + xx_rl).reshape(-1)  # (L,)

	# reverse-upward: reverse both x and y in col-major
	xx_ru, yy_ru = torch.meshgrid(x_rev, y_rev, indexing="ij")  # (W, H)
	order_ru = (yy_ru * w + xx_ru).reshape(-1)  # (L,)

	return order_fl, order_fu, order_rl, order_ru


def bev_to_sequences(z: Tensor, bev_h: int, bev_w: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
	"""Convert BEV feature map to 4 strict 1D sequences.

	Args:
		z: BEV feature tensor of shape (B, H, W, C) or (B, H*W, C).
		bev_h: H
		bev_w: W

	Returns:
		(seq_fl, seq_fu, seq_rl, seq_ru), each is (B, L, C), L=H*W.

	Implementation notes:
		- No zig-zag scanning.
		- No Python for-loop over H/W.
		- Uses index maps (LongTensor) + index_select.
	"""
	z4d = _as_bev_4d(z, bev_h, bev_w)
	b, h, w, c = z4d.shape
	l = h * w

	# Base flatten: row-major (y-major then x)
	flat = z4d.reshape(b, l, c)

	order_fl, order_fu, order_rl, order_ru = _build_order_indices(h, w, device=z4d.device)

	# index_select keeps dtype/device; indices are torch.LongTensor on correct device.
	seq_fl = flat.index_select(dim=1, index=order_fl)
	seq_fu = flat.index_select(dim=1, index=order_fu)
	seq_rl = flat.index_select(dim=1, index=order_rl)
	seq_ru = flat.index_select(dim=1, index=order_ru)

	assert seq_fl.shape == (b, l, c)
	assert seq_fu.shape == (b, l, c)
	assert seq_rl.shape == (b, l, c)
	assert seq_ru.shape == (b, l, c)

	return seq_fl, seq_fu, seq_rl, seq_ru


def _sequence_to_bev_single(seq: Tensor, order: Tensor, bev_h: int, bev_w: int) -> Tensor:
	"""Inverse one sequence back to (B, H, W, C) using scatter.

	Args:
		seq: (B, L, C)
		order: (L,) base indices indicating where each seq position maps in row-major flat.
	"""
	assert isinstance(seq, torch.Tensor), "seq must be a torch.Tensor"
	assert seq.ndim == 3, f"seq must be (B,L,C), got {tuple(seq.shape)}"
	b, l, c = seq.shape
	assert l == bev_h * bev_w, (
		f"seq length L={l} must equal H*W={bev_h}*{bev_w}={bev_h * bev_w}"
	)
	assert order.ndim == 1 and order.numel() == l and order.dtype == torch.long, "order must be LongTensor (L,)"
	assert order.device == seq.device, "order must be on the same device as seq"

	# Scatter seq back to base (row-major) positions.
	out_flat = torch.empty((b, l, c), device=seq.device, dtype=seq.dtype)
	index = order.view(1, l, 1).expand(b, l, c)
	out_flat.scatter_(dim=1, index=index, src=seq)
	return out_flat.reshape(b, bev_h, bev_w, c)


def sequences_to_bev(seqs: Tuple[Tensor, Tensor, Tensor, Tensor], bev_h: int, bev_w: int) -> Tensor:
	"""Re-merge 4 sequences back to BEV (element-wise average of inverses).

	Args:
		seqs: (seq_fl, seq_fu, seq_rl, seq_ru), each is (B, L, C).
		bev_h: H
		bev_w: W

	Returns:
		z: (B, H, W, C), computed as the element-wise average of the four inverse maps.

	Correctness goal:
		sequences_to_bev(bev_to_sequences(Z)) == Z  (exact, up to floating point)
	"""
	assert isinstance(seqs, tuple) and len(seqs) == 4, "seqs must be a tuple of 4 tensors"
	seq_fl, seq_fu, seq_rl, seq_ru = seqs

	# Basic shape/device/dtype checks
	for i, s in enumerate((seq_fl, seq_fu, seq_rl, seq_ru)):
		assert isinstance(s, torch.Tensor), f"seqs[{i}] must be a torch.Tensor"
		assert s.ndim == 3, f"seqs[{i}] must be (B,L,C), got {tuple(s.shape)}"

	b, l, c = seq_fl.shape
	assert seq_fu.shape == (b, l, c)
	assert seq_rl.shape == (b, l, c)
	assert seq_ru.shape == (b, l, c)
	assert l == bev_h * bev_w, (
		f"Sequence length L={l} must equal H*W={bev_h}*{bev_w}={bev_h * bev_w}"
	)
	assert seq_fu.device == seq_fl.device == seq_rl.device == seq_ru.device, "All seqs must be on the same device"
	assert seq_fu.dtype == seq_fl.dtype == seq_rl.dtype == seq_ru.dtype, "All seqs must have the same dtype"

	order_fl, order_fu, order_rl, order_ru = _build_order_indices(bev_h, bev_w, device=seq_fl.device)

	z_fl = _sequence_to_bev_single(seq_fl, order_fl, bev_h, bev_w)
	z_fu = _sequence_to_bev_single(seq_fu, order_fu, bev_h, bev_w)
	z_rl = _sequence_to_bev_single(seq_rl, order_rl, bev_h, bev_w)
	z_ru = _sequence_to_bev_single(seq_ru, order_ru, bev_h, bev_w)

	# Element-wise average of the four inverses.
	z = (z_fl + z_fu + z_rl + z_ru) / 4
	assert z.shape == (b, bev_h, bev_w, c)
	return z

