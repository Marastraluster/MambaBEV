"""Mamba-based DETR3D head for mambaBEV.

This head is a minimal extension over the existing DETR3D / DeformableDETR3D
head logic in MMDet3D. It inserts a single *query sequence modeling* step using
Mamba2 before the deformable attention transformer decoder.

Paper alignment:
- “Mamba-based-DETR”: object queries are processed by Mamba2Block *before*
  the deformable attention decoder, then added back as a residual.

Constraints:
- Reuse base head training/inference logic (loss, coder, assigner, etc.).
- Only add query Mamba-related code; do not modify decoder internals.
"""

from __future__ import annotations

import importlib
from typing import Any, List, Sequence, Tuple, Type

import torch
from torch import nn

from mmdet3d.registry import MODELS

from ..layers.mamba2_wrapper import Mamba2Block


def _resolve_base_head() -> Type[nn.Module]:
	"""Resolve DETR3D-style head base class using importlib candidates.

	This follows the required strategy:
	- Try a list of (module_path, class_name) pairs in order.
	- Use the first successfully imported module that contains the class.
	- If all fail, raise ImportError and list all attempted module paths.
	"""

	candidates: Sequence[Tuple[str, str]] = (
		("mmdet3d.models.dense_heads.detr3d_head", "DETR3DHead"),
		("mmdet3d.models.dense_heads.deformable_detr3d_head", "DeformableDETR3DHead"),
		("projects.DETR3D.detr3d_head", "DETR3DHead"),
		("projects.DETR3D.detr3d_head", "DeformableDETR3DHead"),
		("projects.detr3d.detr3d_head", "DETR3DHead"),
		("projects.detr3d.dense_heads.detr3d_head", "DETR3DHead"),
		("projects.mmdet3d_plugin.detr3d_head", "DETR3DHead"),
	)

	tried: List[str] = []
	last_exc: Exception = ImportError("No candidates tried")

	for module_path, cls_name in candidates:
		tried.append(f"{module_path}:{cls_name}")
		try:
			mod = importlib.import_module(module_path)
		except Exception as exc:
			last_exc = exc
			continue

		if hasattr(mod, cls_name):
			base = getattr(mod, cls_name)
			if isinstance(base, type):
				return base  # type: ignore[return-value]
		# Imported but missing the attribute.
		last_exc = ImportError(f"Module '{module_path}' does not define '{cls_name}'")

	raise ImportError(
		"Failed to import a DETR3D head base class. Tried the following candidates:\n"
		+ "\n".join(tried)
	) from last_exc


BaseHead = _resolve_base_head()


@MODELS.register_module()
class MambaDETR3DHead(BaseHead):
	"""DETR3D head with a pre-decoder Mamba2 query block."""

	def __init__(
		self,
		num_query: int = 900,
		embed_dims: int = 256,
		mamba_dim: int = 256,
		d_state: int = 128,
		d_conv: int = 4,
		expand: int = 2,
		**kwargs: Any,
	) -> None:
		# Pass-through all base args via **kwargs while enforcing defaults.
		if "num_query" not in kwargs:
			kwargs["num_query"] = num_query

		# Keep naming consistent with most DETR3D-style heads.
		if "embed_dims" not in kwargs and "hidden_dim" not in kwargs and "embed_dim" not in kwargs:
			kwargs["embed_dims"] = embed_dims

		super().__init__(**kwargs)

		# Base head should expose embed_dims; ensure consistency for our Linear layers.
		base_embed_dims = getattr(self, "embed_dims", embed_dims)
		assert isinstance(base_embed_dims, int) and base_embed_dims > 0
		assert (
			base_embed_dims == embed_dims
		), f"Base head embed_dims={base_embed_dims} differs from requested embed_dims={embed_dims}"

		# (仅新增) Query Mamba 组件：dropout=0.0
		self.query_in_proj = nn.Linear(embed_dims, mamba_dim)
		self.query_mamba = Mamba2Block(
			d_model=mamba_dim,
			d_state=d_state,
			d_conv=d_conv,
			expand=expand,
			dropout=0.0,
		)
		self.query_out_proj = nn.Linear(mamba_dim, embed_dims)

	def _apply_query_mamba(self, query_embed: torch.Tensor) -> torch.Tensor:
		"""Mamba-based-DETR: apply Mamba2 on object queries before deformable attention decoder.

		This is the ONLY intended logic change vs the base head.

		Expected query embedding layouts:
		- (num_query, 2*embed_dims): split into (query_pos, query)
		- (num_query, embed_dims): only query

		We only apply Mamba to `query` (not query_pos), then add residual.
		"""
		assert isinstance(query_embed, torch.Tensor), "query_embed must be a torch.Tensor"
		assert query_embed.ndim == 2, (
			f"query_embed must be 2D (num_query, D), got shape={tuple(query_embed.shape)}"
		)

		c = getattr(self, "embed_dims", None)
		assert isinstance(c, int) and c > 0, "Base head must expose embed_dims"

		if query_embed.shape[1] == 2 * c:
			query_pos, query = torch.split(query_embed, c, dim=1)
			# Mamba2Block expects (B, L, C); treat batch as 1 and L=num_query.
			q2 = self.query_out_proj(
				self.query_mamba(self.query_in_proj(query.unsqueeze(0))).squeeze(0)
			)
			query = query + q2
			return torch.cat([query_pos, query], dim=1)

		if query_embed.shape[1] == c:
			q2 = self.query_out_proj(
				self.query_mamba(self.query_in_proj(query_embed.unsqueeze(0))).squeeze(0)
			)
			return query_embed + q2

		raise AssertionError(
			f"Unsupported query_embed dim={query_embed.shape[1]} for embed_dims={c}. "
			"Expected (num_query, 2*embed_dims) or (num_query, embed_dims)."
		)

	def forward(self, mlvl_feats, img_metas, **kwargs):
		"""Forward function.

		This mirrors the base DETR3D head forward as closely as possible.
		The ONLY difference is that we apply a pre-decoder query Mamba residual
		on object queries.
		"""

		if not hasattr(self, "query_embedding"):
			raise AttributeError(
				"Base head does not have 'query_embedding'. "
				"Please verify the DETR3D head implementation in your mmdet3d version."
			)

		# ---唯一允许的逻辑改动点---
		query_embeds = self._apply_query_mamba(self.query_embedding.weight)
		# ---其余逻辑完全复用/对齐基类---

		hs, init_reference, inter_references = self.transformer(
			mlvl_feats,
			query_embeds,
			reg_branches=self.reg_branches if getattr(self, "with_box_refine", False) else None,
			img_metas=img_metas,
			**kwargs,
		)

		# Keep the base post-processing.
		from mmdet.models.layers import inverse_sigmoid  # type: ignore

		hs = hs.permute(0, 2, 1, 3)
		outputs_classes = []
		outputs_coords = []

		pc_range = getattr(self, "pc_range", None)
		assert pc_range is not None, "Base head must define pc_range"

		for lvl in range(hs.shape[0]):
			reference = init_reference if lvl == 0 else inter_references[lvl - 1]
			reference = inverse_sigmoid(reference)
			outputs_class = self.cls_branches[lvl](hs[lvl])
			tmp = self.reg_branches[lvl](hs[lvl])
			assert reference.shape[-1] == 3

			tmp[..., 0:2] += reference[..., 0:2]
			tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
			tmp[..., 4:5] += reference[..., 2:3]
			tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

			tmp[..., 0:1] = tmp[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
			tmp[..., 1:2] = tmp[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
			tmp[..., 4:5] = tmp[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2]

			outputs_classes.append(outputs_class)
			outputs_coords.append(tmp)

		outputs_classes = torch.stack(outputs_classes)
		outputs_coords = torch.stack(outputs_coords)

		return {
			"all_cls_scores": outputs_classes,
			"all_bbox_preds": outputs_coords,
			"enc_cls_scores": None,
			"enc_bbox_preds": None,
		}

