"""MambaBEVDetector.

This detector is a minimal-intrusion wrapper that inserts TemporalMamba fusion
*after single-frame BEV is produced and before it is consumed by the head*.

Key constraints:
- Do not modify head/decoder/loss/coder/assigner/dataset pipeline.
- Only insert TemporalMamba once: after curr_bev, before head.
- Robustly degrade when prev_bev / delta_yaw are unavailable.

Implementation strategy:
- Dynamically locate a suitable BaseDetector via importlib candidates.
- Override `extract_feat` (the most common hook used by MMDet3D detectors)
  to detect a BEV tensor in the returned structure and fuse it in-place.
  This keeps the rest of the base detector's forward/loss/predict logic intact.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn

from mmdet3d.registry import MODELS

from ..layers.temporal_mamba import TemporalMamba


Tensor = torch.Tensor


def _resolve_base_detector() -> Type[nn.Module]:
	"""Resolve a detector base class using importlib candidates.

	Required candidate list (in order):
	  1) mmdet3d.models.detectors.bevformer: BEVFormer
	  2) mmdet3d.models.detectors.bevformer: BEVFormerDetector
	  3) mmdet3d.models.detectors.detr3d: DETR3D
	  4) mmdet3d.models.detectors.detr3d: DETR3DDetector
	  5) projects.BEVFormer.bevformer: BEVFormer
	  6) projects.bevformer.bevformer: BEVFormer
	  7) projects.DETR3D.detr3d: DETR3D
	  8) projects.detr3d.detr3d: DETR3D
	  9) projects.mmdet3d_plugin.bevformer: BEVFormer
	 10) projects.mmdet3d_plugin.detr3d: DETR3D
	"""

	candidates: Sequence[Tuple[str, str]] = (
		("mmdet3d.models.detectors.bevformer", "BEVFormer"),
		("mmdet3d.models.detectors.bevformer", "BEVFormerDetector"),
		("mmdet3d.models.detectors.detr3d", "DETR3D"),
		("mmdet3d.models.detectors.detr3d", "DETR3DDetector"),
		("projects.BEVFormer.bevformer", "BEVFormer"),
		("projects.bevformer.bevformer", "BEVFormer"),
		("projects.DETR3D.detr3d", "DETR3D"),
		("projects.detr3d.detr3d", "DETR3D"),
		("projects.mmdet3d_plugin.bevformer", "BEVFormer"),
		("projects.mmdet3d_plugin.detr3d", "DETR3D"),
	)

	tried: List[str] = []
	errors: List[str] = []
	for module_path, cls_name in candidates:
		tried.append(f"{module_path}:{cls_name}")
		try:
			mod = importlib.import_module(module_path)
		except Exception as exc:
			errors.append(f"- {module_path}: import failed: {type(exc).__name__}: {exc}")
			continue
		if not hasattr(mod, cls_name):
			errors.append(f"- {module_path}: missing attribute '{cls_name}'")
			continue
		cls = getattr(mod, cls_name)
		if isinstance(cls, type):
			return cls  # type: ignore[return-value]
		errors.append(f"- {module_path}: '{cls_name}' is not a class")

	raise ImportError(
		"Failed to resolve BaseDetector for MambaBEVDetector. Tried candidates:\n"
		+ "\n".join(tried)
		+ "\n\nErrors:\n"
		+ "\n".join(errors)
	)


BaseDetector = _resolve_base_detector()


def _get_meta_list_from_data_samples(batch_data_samples: Any) -> Optional[List[dict]]:
	"""Best-effort extraction of metainfo dict list from data_samples."""
	if batch_data_samples is None:
		return None
	if isinstance(batch_data_samples, (list, tuple)):
		metas: List[dict] = []
		for item in batch_data_samples:
			if hasattr(item, "metainfo"):
				metas.append(getattr(item, "metainfo"))
			elif hasattr(item, "metainfo") is False and hasattr(item, "metainfo"):
				metas.append(getattr(item, "metainfo"))
			elif isinstance(item, dict):
				metas.append(item)
			else:
				# Unknown sample type
				return None
		return metas
	return None


def _dict_like_get(obj: Any, key: str, default: Any = None) -> Any:
	"""Get from dict-like object (including mmengine InstanceData-like)."""
	if obj is None:
		return default
	if isinstance(obj, Mapping):
		return obj.get(key, default)
	if hasattr(obj, "get"):
		try:
			return obj.get(key, default)
		except Exception:
			return default
	return default


def _try_parse_delta_yaw_from_can_bus(can_bus: Any) -> Optional[Union[float, Tensor]]:
	"""Best-effort parse of delta_yaw from can_bus without assuming indices.

	We avoid hard-coded indices because different projects use different layouts.
	Supported patterns:
	  - dict-like: {'delta_yaw': ...}
	  - dict-like: {'yaw': ..., 'prev_yaw': ...} -> yaw - prev_yaw

	If ambiguous, return None.
	"""
	if can_bus is None:
		return None
	if isinstance(can_bus, Mapping):
		dy = can_bus.get("delta_yaw", None)
		if dy is not None:
			return dy
		yaw = can_bus.get("yaw", None)
		prev_yaw = can_bus.get("prev_yaw", None)
		if yaw is not None and prev_yaw is not None:
			try:
				return float(yaw) - float(prev_yaw)
			except Exception:
				return None
	return None


def _stack_optional_floats(values: List[Any], device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
	"""Convert a per-sample list of yaw values into a (B,) tensor if possible."""
	if not values:
		return None
	if all(v is None for v in values):
		return None
	if any(v is None for v in values):
		return None
	try:
		return torch.tensor([float(v) for v in values], device=device, dtype=dtype)
	except Exception:
		return None


@MODELS.register_module()
class MambaBEVDetector(BaseDetector):
	"""Detector that inserts TemporalMamba fusion before head.

	TemporalMamba fusion inserted here: after single-frame BEV, before head.
	"""

	def __init__(
		self,
		*args: Any,
		temporal_mamba: Optional[Union[Dict[str, Any], nn.Module]] = None,
		**kwargs: Any,
	) -> None:
		# Forward all base params untouched.
		super().__init__(*args, **kwargs)

		# Build / attach TemporalMamba.
		if temporal_mamba is None:
			# Default: keep minimal hardcoding; allow config override by passing temporal_mamba.
			self.temporal_mamba = TemporalMamba(embed_dim=256, mamba_dim=256, dropout=0.9)
		elif isinstance(temporal_mamba, nn.Module):
			self.temporal_mamba = temporal_mamba
		elif isinstance(temporal_mamba, dict):
			# Try mmengine-style build first; fall back to direct instantiation.
			cfg = dict(temporal_mamba)
			try:
				self.temporal_mamba = MODELS.build(cfg)  # type: ignore[assignment]
			except Exception:
				t = cfg.pop("type", None)
				if t in (None, "TemporalMamba"):
					self.temporal_mamba = TemporalMamba(**cfg)
				else:
					raise
		else:
			raise TypeError("temporal_mamba must be a dict, nn.Module, or None")

	def _get_bev_hw(
		self,
		curr_bev: Tensor,
		batch_input_metas: Optional[List[dict]] = None,
		**kwargs: Any,
	) -> Tuple[int, int]:
		"""Resolve bev_h, bev_w without guessing.

		Priority:
		  - explicit kwargs
		  - self.bev_h/self.bev_w
		  - meta fields ('bev_h','bev_w') or ('bev_shape')
		"""
		bev_h = kwargs.get("bev_h", None)
		bev_w = kwargs.get("bev_w", None)

		if bev_h is None and hasattr(self, "bev_h"):
			bev_h = getattr(self, "bev_h")
		if bev_w is None and hasattr(self, "bev_w"):
			bev_w = getattr(self, "bev_w")

		if (bev_h is None or bev_w is None) and batch_input_metas:
			m0 = batch_input_metas[0]
			bev_h = m0.get("bev_h", bev_h)
			bev_w = m0.get("bev_w", bev_w)
			if (bev_h is None or bev_w is None) and "bev_shape" in m0:
				try:
					shape = m0["bev_shape"]
					# Accept (H, W) or (H, W, C)
					bev_h = int(shape[0])
					bev_w = int(shape[1])
				except Exception:
					pass

		assert isinstance(bev_h, int) and isinstance(bev_w, int) and bev_h > 0 and bev_w > 0, (
			"bev_h/bev_w must be provided by config/metadata (cannot be inferred). "
			"Pass bev_h/bev_w via kwargs or set meta['bev_h'/'bev_w']."
		)

		b, l, c = curr_bev.shape
		assert bev_h * bev_w == l, f"curr_bev L must equal H*W: got L={l}, H*W={bev_h}*{bev_w}={bev_h * bev_w}"
		return bev_h, bev_w

	def _resolve_prev_bev(
		self,
		batch_input_metas: Optional[List[dict]] = None,
		batch_data_samples: Any = None,
		**kwargs: Any,
	) -> Optional[Tensor]:
		"""Resolve prev_bev with robust degradation."""
		# (a) direct inputs
		prev = kwargs.get("prev_bev", None)
		if isinstance(prev, torch.Tensor):
			return prev

		# (b) from metas or data_samples
		if batch_input_metas:
			candidate = batch_input_metas[0].get("prev_bev", None)
			if isinstance(candidate, torch.Tensor):
				return candidate

		if isinstance(batch_data_samples, (list, tuple)) and len(batch_data_samples) > 0:
			candidate = _dict_like_get(batch_data_samples[0], "prev_bev", None)
			if isinstance(candidate, torch.Tensor):
				return candidate

		return None

	def _resolve_delta_yaw(
		self,
		device: torch.device,
		dtype: torch.dtype,
		batch_input_metas: Optional[List[dict]] = None,
		batch_data_samples: Any = None,
		**kwargs: Any,
	) -> Optional[Tensor]:
		"""Resolve delta_yaw with robust degradation.

		Priority:
		  a) parse from img_metas[i]['can_bus'] if possible (no hard-coded indices)
		  b) img_metas[i].get('delta_yaw')
		  c) data_samples[i].get('delta_yaw')
		  d) None
		"""

		# Direct override via kwargs
		dy = kwargs.get("delta_yaw", None)
		if isinstance(dy, torch.Tensor):
			if dy.ndim == 0:
				return dy.reshape(1).to(device=device, dtype=dtype)
			return dy.to(device=device, dtype=dtype)

		if batch_input_metas:
			# per-sample parse
			parsed: List[Any] = []
			for meta in batch_input_metas:
				can_bus = meta.get("can_bus", None)
				v = _try_parse_delta_yaw_from_can_bus(can_bus)
				if v is None:
					v = meta.get("delta_yaw", None)
				parsed.append(v)
			out = _stack_optional_floats(parsed, device=device, dtype=dtype)
			if out is not None:
				return out

		if isinstance(batch_data_samples, (list, tuple)) and len(batch_data_samples) > 0:
			parsed2: List[Any] = []
			for sample in batch_data_samples:
				parsed2.append(_dict_like_get(sample, "delta_yaw", None))
			out = _stack_optional_floats(parsed2, device=device, dtype=dtype)
			if out is not None:
				return out

		return None

	def _fuse_curr_bev(
		self,
		curr_bev: Tensor,
		batch_input_metas: Optional[List[dict]] = None,
		batch_data_samples: Any = None,
		**kwargs: Any,
	) -> Tensor:
		"""Apply TemporalMamba on curr_bev if possible."""
		assert curr_bev.ndim == 3, f"curr_bev must be (B,L,C), got {tuple(curr_bev.shape)}"
		b, l, c = curr_bev.shape

		bev_h, bev_w = self._get_bev_hw(curr_bev, batch_input_metas=batch_input_metas, **kwargs)

		prev_bev = self._resolve_prev_bev(batch_input_metas=batch_input_metas, batch_data_samples=batch_data_samples, **kwargs)
		if prev_bev is not None:
			assert prev_bev.shape == curr_bev.shape, (
				f"prev_bev must match curr_bev shape: got {tuple(prev_bev.shape)} vs {tuple(curr_bev.shape)}"
			)

		delta_yaw = self._resolve_delta_yaw(
			device=curr_bev.device,
			dtype=curr_bev.dtype,
			batch_input_metas=batch_input_metas,
			batch_data_samples=batch_data_samples,
			**kwargs,
		)

		# TemporalMamba fusion inserted here: after single-frame BEV, before head
		fused = self.temporal_mamba(
			curr_bev=curr_bev,
			prev_bev=prev_bev,
			bev_h=bev_h,
			bev_w=bev_w,
			delta_yaw=delta_yaw,
		)

		assert fused.shape == curr_bev.shape, (
			f"TemporalMamba must return same shape as curr_bev: got {tuple(fused.shape)} vs {tuple(curr_bev.shape)}"
		)
		return fused

	def extract_feat(self, batch_inputs_dict: Dict[str, Any], batch_input_metas: Optional[List[dict]] = None, **kwargs: Any):
		"""Override extract_feat to fuse BEV features with TemporalMamba.

		This keeps the base detector's loss/predict logic unchanged while
		inserting fusion at the earliest stable point where BEV is available.
		"""
		feats = super().extract_feat(batch_inputs_dict, batch_input_metas, **kwargs)

		# Try to locate a BEV tensor in the returned structure.
		# Common patterns:
		#   - Tensor: curr_bev itself
		#   - dict: {'bev': ..., 'bev_feat': ..., 'bev_embed': ..., 'curr_bev': ...}
		#   - tuple/list: (img_feats, curr_bev, ...)

		# Helper to decide whether a tensor is a BEV (B,L,C).
		def _is_bev_tensor(t: Any) -> bool:
			return isinstance(t, torch.Tensor) and t.ndim == 3

		if _is_bev_tensor(feats):
			return self._fuse_curr_bev(feats, batch_input_metas=batch_input_metas, **kwargs)

		if isinstance(feats, dict):
			for k in ("curr_bev", "bev", "bev_feat", "bev_embed"):
				if k in feats and _is_bev_tensor(feats[k]):
					feats[k] = self._fuse_curr_bev(
						feats[k],
						batch_input_metas=batch_input_metas,
						batch_data_samples=kwargs.get("batch_data_samples", None),
						**kwargs,
					)
					return feats
			return feats

		if isinstance(feats, (list, tuple)):
			# Find first (B,L,C) tensor and replace.
			replaced = False
			out_list = list(feats)
			for i, item in enumerate(out_list):
				if _is_bev_tensor(item):
					out_list[i] = self._fuse_curr_bev(
						item,
						batch_input_metas=batch_input_metas,
						batch_data_samples=kwargs.get("batch_data_samples", None),
						**kwargs,
					)
					replaced = True
					break
			if replaced:
				return type(feats)(out_list)  # preserve list/tuple
			return feats

		# Unknown structure; do not modify.
		return feats

