"""Mamba2 wrapper.

This module provides a *sequence-level Mamba block* intended for modeling sequences
such as BEV features and object queries.

Constraints:
- Depends only on torch and mamba-ssm.
- No mmdet/mmengine dependencies.
- No explicit CUDA extension compilation steps.
"""

from __future__ import annotations

from typing import Any, Type

import torch
from torch import nn


def _resolve_mamba_impl() -> Type[nn.Module]:
	"""Resolve an available Mamba implementation.

	Tries (in order):
	  1) mamba_ssm.modules.mamba2.Mamba2
	  2) mamba_ssm.Mamba

	Returns:
		A torch.nn.Module class implementing a Mamba-like layer.

	Raises:
		ImportError: If neither implementation can be imported.
	"""

	try:
		# Preferred in newer versions (including mamba-ssm==2.2.2)
		from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore

		return Mamba2
	except Exception:
		pass

	try:
		# Fallback import path used by some variants/releases
		from mamba_ssm import Mamba  # type: ignore

		return Mamba
	except Exception as exc:
		raise ImportError("Please install mamba-ssm==2.2.2") from exc


class Mamba2Block(nn.Module):
	"""A robust sequence-level Mamba block.

	Input/Output:
		- x: (B, L, C)
		- y: (B, L, C)

	Notes:
		- Preserves dtype (fp16/fp32) and does not enforce autocast.
		- Applies dropout after the Mamba output.
	"""

	def __init__(
		self,
		d_model: int,
		d_state: int = 128,
		d_conv: int = 4,
		expand: int = 2,
		dropout: float = 0.0,
	) -> None:
		super().__init__()

		assert isinstance(d_model, int) and d_model > 0, "d_model must be a positive int"
		assert isinstance(d_state, int) and d_state > 0, "d_state must be a positive int"
		assert isinstance(d_conv, int) and d_conv > 0, "d_conv must be a positive int"
		assert isinstance(expand, int) and expand > 0, "expand must be a positive int"
		assert isinstance(dropout, float) and 0.0 <= dropout <= 1.0, "dropout must be in [0, 1]"

		self.d_model = d_model
		self.d_state = d_state
		self.d_conv = d_conv
		self.expand = expand

		MambaImpl = _resolve_mamba_impl()
		self.mamba = self._build_mamba(MambaImpl)
		self.dropout = nn.Dropout(p=dropout)

	def _build_mamba(self, impl: Type[nn.Module]) -> nn.Module:
		"""Instantiate the selected Mamba implementation.

		Different versions may accept either keyword-only or positional `d_model`.
		We try a small set of compatible constructor patterns.
		"""

		kwargs: dict[str, Any] = {
			"d_model": self.d_model,
			"d_state": self.d_state,
			"d_conv": self.d_conv,
			"expand": self.expand,
		}

		# 1) Most common: keyword arguments
		try:
			return impl(**kwargs)
		except TypeError:
			pass

		# 2) Positional d_model + keyword rest
		try:
			return impl(self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
		except TypeError:
			pass

		# 3) Pure positional (least preferred)
		try:
			return impl(self.d_model, self.d_state, self.d_conv, self.expand)
		except TypeError as exc:
			raise TypeError(
				f"Failed to construct {impl.__name__} with d_model={self.d_model}, "
				f"d_state={self.d_state}, d_conv={self.d_conv}, expand={self.expand}."
			) from exc

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
		assert x.ndim == 3, f"x must be a 3D tensor with shape (B, L, C), got {tuple(x.shape)}"
		assert (
			x.shape[-1] == self.d_model
		), f"x last dim (C) must equal d_model={self.d_model}, got C={x.shape[-1]}"

		# Mamba implementation is expected to accept (B, L, C) and return (B, L, C).
		y = self.mamba(x)
		y = self.dropout(y)

		assert (
			y.shape == x.shape
		), f"Output shape must match input shape; got {tuple(y.shape)} vs {tuple(x.shape)}"
		return y

