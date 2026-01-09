"""Ego-motion alignment utilities for BEV features.

This file provides a yaw-rotation alignment tool for historical BEV features.

Goal:
- Rotate the previous-frame BEV feature by `delta_yaw` to align it into the
  current-frame coordinate system.

Implementation:
- Uses torch.nn.functional.affine_grid + grid_sample.
- No external libraries.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F


def align_bev_by_yaw(
	bev: torch.Tensor,  # (B, H, W, C)
	delta_yaw: Union[torch.Tensor, float],  # (B,) or float
	mode: str = "bilinear",
	padding_mode: str = "zeros",
	align_corners: bool = True,
) -> torch.Tensor:
	"""Rotate BEV feature around its center to align by yaw.

	Args:
		bev: Tensor of shape (B, H, W, C).
		delta_yaw: CCW-positive yaw (radians). Either a tensor of shape (B,)
			or a python float (broadcast to the batch).
		mode: Interpolation mode for grid_sample.
		padding_mode: Padding mode for grid_sample. Default "zeros" matches
			the common choice of filling empty locations with 0.
		align_corners: Passed to affine_grid/grid_sample.

	Returns:
		Rotated BEV tensor with the same shape (B, H, W, C).

	Notes on coordinates and signs:
		- grid_sample uses normalized coordinates in [-1, 1] with x to the right
		  and y *downward* (image convention).
		- The input `delta_yaw` is defined as CCW-positive in the usual 2D plane
		  (x right, y up). To respect that while operating in image coordinates
		  (y down), we use an effective rotation matrix: F * R(-yaw) * F,
		  where F = diag(1, -1) flips the y-axis.
		- We use -delta_yaw because affine_grid maps output coordinates -> input
		  coordinates (inverse warping).
	"""

	assert isinstance(bev, torch.Tensor), "bev must be a torch.Tensor"
	assert bev.ndim == 4, f"bev must be 4D (B,H,W,C), got shape={tuple(bev.shape)}"
	assert bev.is_floating_point(), "bev must be a floating point tensor (fp16/fp32)"

	b, h, w, c = bev.shape
	assert b > 0 and h > 0 and w > 0 and c > 0, f"Invalid bev shape={tuple(bev.shape)}"

	if isinstance(delta_yaw, float):
		yaw = torch.full((b,), float(delta_yaw), device=bev.device, dtype=bev.dtype)
	else:
		assert isinstance(delta_yaw, torch.Tensor), "delta_yaw must be a torch.Tensor or float"
		if delta_yaw.ndim == 0:
			yaw = delta_yaw.reshape(1).to(device=bev.device, dtype=bev.dtype).expand(b)
		else:
			assert delta_yaw.ndim == 1, f"delta_yaw must be shape (B,), got {tuple(delta_yaw.shape)}"
			assert (
				delta_yaw.shape[0] == b
			), f"delta_yaw batch size must match bev: got {delta_yaw.shape[0]} vs {b}"
			yaw = delta_yaw.to(device=bev.device, dtype=bev.dtype)

	# affine_grid produces a grid for sampling the input.
	# To rotate the content by +yaw (CCW), we sample using the inverse transform:
	# p_in = R(-yaw) p_out.
	a = -yaw
	cos_a = torch.cos(a)
	sin_a = torch.sin(a)

	# Effective rotation in grid_sample's (x right, y down) coordinates that
	# corresponds to CCW rotation in a standard (x right, y up) plane:
	# M = F * R(a) * F = [[cos,  sin],
	#                     [-sin, cos]]
	# Here a = -yaw (inverse warp).
	zeros = torch.zeros_like(cos_a)
	row0 = torch.stack([cos_a, sin_a, zeros], dim=-1)      # (B, 3)
	row1 = torch.stack([-sin_a, cos_a, zeros], dim=-1)     # (B, 3)
	theta = torch.stack([row0, row1], dim=1)               # (B, 2, 3)

	# grid_sample expects NCHW
	bev_in = bev.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
	grid = F.affine_grid(theta, size=bev_in.shape, align_corners=align_corners)
	out = F.grid_sample(
		bev_in,
		grid,
		mode=mode,
		padding_mode=padding_mode,
		align_corners=align_corners,
	)
	bev_out = out.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

	assert (
		bev_out.shape == bev.shape
	), f"Output shape must match input: got {tuple(bev_out.shape)} vs {tuple(bev.shape)}"
	return bev_out

