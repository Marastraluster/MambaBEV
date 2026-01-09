"""mambaBEV Base (R101, 1600x900, 4 history frames).

Paper-aligned Base setting for mambaBEV.
Based on single-frame DETR3D (ResNet-101).
"""

# ========================
# 1) Base inheritance
# ========================
# IMPORTANT:
# Use single-frame DETR3D with ResNet-101.
# If your repo provides a DCN variant (R101-DCN / R101-CDN),
# that is consistent with the paper and can be used here.
_base_ = [
    "../../mmdetection3d/projects/DETR3D/configs/detr3d_r101.py",
]

# ========================
# 2) custom_imports (required)
# ========================
custom_imports = dict(
    imports=["mambabev"],
    allow_failed_imports=False,
)

# Explicitly trigger registration
import mambabev  # noqa: E402
mambabev.register_all_modules()  # noqa: E402

# ========================
# 3) Dataset / class names
# ========================
# nuScenes standard 10 classes
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# ========================
# 4) Base-specific settings (paper-aligned)
# ========================
img_scale = (1600, 900)

bev_h = 200
bev_w = 200
bev_resolution = 0.512

history_frames = 4
encoder_layers = 6

embed_dims = 256

# Put BEV shape into metainfo so detector can read it safely
metainfo = dict(
    classes=class_names,
    bev_h=bev_h,
    bev_w=bev_w,
    bev_resolution=bev_resolution,
    history_frames=history_frames,
)

# ========================
# 5) Data pipeline (NO extra augmentation)
# ========================
# Deterministic resize only (paper: no data augmentation)
train_pipeline = [
    dict(
        type="RandomResize3D",
        scale=img_scale,
        ratio_range=(1.0, 1.0),
        keep_ratio=True,
    ),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        metainfo=metainfo,
    )
)

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

test_dataloader = val_dataloader

# ========================
# 6) Model (core)
# ========================
model = dict(
    type="MambaBEVDetector",

    # Explicit BEV shape (robust fallback)
    bev_h=bev_h,
    bev_w=bev_w,

    temporal_mamba=dict(
        type="TemporalMamba",
        embed_dim=embed_dims,
        mamba_dim=embed_dims,
        dropout=0.9,  # paper-specified residual dropout
    ),

    # Backbone: ResNet-101 (Base)
    # If your base config already defines DCN/CDN, keep it.
    img_backbone=dict(
        _delete_=True,
        type="mmdet.ResNet",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
    ),

    # Transformer: explicitly set encoder layers (Base = 6)
    transformer=dict(
        encoder=dict(
            num_layers=encoder_layers,
        )
    ),

    # Head: Mamba-based DETR3D
    pts_bbox_head=dict(
        type="MambaDETR3DHead",
        num_query=900,          # paper
        embed_dims=embed_dims,
        mamba_dim=embed_dims,
        d_state=128,
        d_conv=4,
        expand=2,
    ),
)

# ========================
# 7) Optimizer (strict paper setting)
# ========================
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
)

# ========================
# 8) Training / evaluation
# ========================
# Keep epoch schedule, hooks, evaluators from base config
# nuScenes metrics: mAP / NDS

# ========================
# 9) Work directory
# ========================
work_dir = "./work_dirs/mambabev_base_r101_1600x900_4f"
